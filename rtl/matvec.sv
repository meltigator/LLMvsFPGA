// LLMvsFPGA - INT8 matvec accelerator, M1 version.
//
// Avalon-MM slave (word addressing, fixed read latency 1, no waitrequest).
// Register map and behavior: docs/architecture.md, sections 7-8.
//
// M1 changes with respect to M0:
//   * W and X live in block RAM (M10K-friendly: simple dual port, synchronous
//     read, byte-wide write lanes), organized LANES bytes wide so that one
//     row-slice is fetched per cycle.
//   * Fully pipelined datapath: RAM -> multipliers -> registered adder tree
//     -> accumulator -> Y RAM. One LANES-wide slice enters every cycle.
//   * No variable divider: GROUP_SIZE must be 0 or a power of two >= LANES.
//   * W layout: each row is padded to a multiple of LANES bytes (see below).
//
// Memory layouts (little-endian, byte 0 = bits [7:0] of the 32-bit word):
//   W window : int8, row r starts at byte r * CPR * LANES, CPR = ceil(DIM_N/LANES)
//              (identical to packed row-major when DIM_N is a multiple of LANES;
//              padding bytes may hold anything, they are masked)
//   X window : int8, index = col        (readable only while not BUSY)
//   Y window : int32, index = row*groups + group
//
// Timing: CYCLES = DIM_M * CPR + PIPE_LATENCY,  PIPE_LATENCY = log2(LANES) + 3.

module matvec #(
    parameter LANES   = 16,     // power of two, >= 4
    parameter M_MAX   = 64,     // max rows
    parameter N_MAX   = 1024,   // max columns
    parameter Y_WORDS = 2048    // Y window capacity in 32-bit words
) (
    input  wire        clk,
    input  wire        reset_n,

    input  wire [14:0] avs_address,      // WORD address (byte offset >> 2)
    input  wire        avs_read,
    input  wire        avs_write,
    input  wire [31:0] avs_writedata,
    input  wire [3:0]  avs_byteenable,
    output wire [31:0] avs_readdata      // valid one cycle after avs_read
);

    // ------------------------------------------------------------ constants
    localparam LG       = $clog2(LANES);              // adder-tree depth
    localparam WB       = LANES / 4;                  // 32-bit RAM banks per wide word
    localparam CPR_MAX  = (N_MAX + LANES - 1) / LANES;
    localparam W_DEPTH  = M_MAX * CPR_MAX;            // wide words in W RAM
    localparam X_DEPTH  = CPR_MAX;                    // wide words in X RAM
    localparam ACCW     = 16 + LG;                    // width of a tree sum
    localparam AW_W     = (W_DEPTH > 1) ? $clog2(W_DEPTH) : 1;
    localparam AW_X     = (X_DEPTH > 1) ? $clog2(X_DEPTH) : 1;
    localparam AW_Y     = (Y_WORDS > 1) ? $clog2(Y_WORDS) : 1;

    localparam [31:0] ID_VALUE      = 32'h4C4C4650;   // "LLFP"
    localparam [31:0] VERSION_VALUE = 32'h00000002;   // M1
    localparam [15:0] M_MAX_16      = M_MAX;
    localparam [15:0] N_MAX_16      = N_MAX;

    initial begin
        if (LANES < 4 || (LANES & (LANES - 1)) != 0) $error("LANES must be a power of two >= 4");
        if (W_DEPTH * LANES > 65536)  $error("W capacity exceeds the 64 KB W window");
        if (N_MAX > 4096)             $error("N_MAX exceeds the 4 KB X window");
        if (Y_WORDS > 2048)           $error("Y_WORDS exceeds the 8 KB Y window");
        if (N_MAX >= 65536)           $error("N_MAX must fit in 16 bits");
    end

    // ------------------------------------------------------------ registers
    reg [31:0] dim_m, dim_n, gsize, cycles;
    reg        busy, done, err;

    // configuration latched at START
    reg [31:0] cfg_cpr_m1;     // chunks per row - 1
    reg [31:0] cfg_gch_m1;     // chunks per group - 1
    reg [31:0] cfg_rows_m1;    // rows - 1
    reg [31:0] cfg_tail;       // valid lanes in the last chunk of a row (1..LANES)
    reg [31:0] cfg_ytotal;     // number of results

    // issue-stage state
    reg        issue_active;
    reg [31:0] row, c, gcnt, waddr;

    // ------------------------------------------------------------ decode
    wire ctrl_sel = (avs_address[14:6]  == 9'd0);      // byte 0x0000-0x00FF
    wire x_sel    = (avs_address[14:10] == 5'b00001);  // byte 0x1000-0x1FFF
    wire y_sel    = (avs_address[14:11] == 4'b0001);   // byte 0x2000-0x3FFF
    wire w_sel    = avs_address[14];                   // byte 0x10000-0x1FFFF

    wire [13:0] w_widx  = avs_address[13:0];
    wire [9:0]  x_widx  = avs_address[9:0];
    wire [31:0] w_bank  = w_widx % WB;
    wire [31:0] w_baddr = w_widx / WB;
    wire [31:0] x_bank  = x_widx % WB;
    wire [31:0] x_baddr = x_widx / WB;

    wire bus_w_we = avs_write && w_sel && !busy && (w_baddr < W_DEPTH);
    wire bus_x_we = avs_write && x_sel && !busy && (x_baddr < X_DEPTH);

    // ------------------------------------------------------------ START validation
    function integer log2_of;                 // position of the highest set bit
        input [31:0] v;
        integer i;
        begin
            log2_of = 0;
            for (i = 0; i < 32; i = i + 1)
                if (v[i]) log2_of = i;
        end
    endfunction

    wire        gs_pow2  = (gsize != 32'd0) && ((gsize & (gsize - 32'd1)) == 32'd0);
    wire [31:0] cpr      = (dim_n + LANES - 1) >> LG;
    wire [31:0] gch      = (gsize == 32'd0) ? cpr : (gsize >> LG);
    wire [31:0] groups_per_row = (gsize == 32'd0) ? 32'd1 : (dim_n >> log2_of(gsize));
    wire [31:0] y_total  = dim_m[15:0] * groups_per_row[15:0];

    wire cfg_ok =
           (dim_m >= 32'd1) && (dim_m <= M_MAX)
        && (dim_n >= 32'd1) && (dim_n <= N_MAX)
        && ((gsize == 32'd0)
            || (gs_pow2 && (gsize >= LANES) && (gsize <= dim_n)
                && ((dim_n & (gsize - 32'd1)) == 32'd0)))
        && (y_total <= Y_WORDS);

    // ------------------------------------------------------------ W RAM (write: bus, read: core)
    wire [AW_W-1:0]      w_raddr = waddr[AW_W-1:0];
    wire [LANES*8-1:0]   w_data;

    genvar gb;
    generate
        for (gb = 0; gb < WB; gb = gb + 1) begin : g_wb
            (* ramstyle = "M10K, no_rw_check" *) reg [7:0] m0 [0:W_DEPTH-1];
            (* ramstyle = "M10K, no_rw_check" *) reg [7:0] m1 [0:W_DEPTH-1];
            (* ramstyle = "M10K, no_rw_check" *) reg [7:0] m2 [0:W_DEPTH-1];
            (* ramstyle = "M10K, no_rw_check" *) reg [7:0] m3 [0:W_DEPTH-1];
            reg [31:0] rd;
            wire we = bus_w_we && (w_bank == gb);
            always @(posedge clk) begin
                if (we && avs_byteenable[0]) m0[w_baddr[AW_W-1:0]] <= avs_writedata[ 7: 0];
                if (we && avs_byteenable[1]) m1[w_baddr[AW_W-1:0]] <= avs_writedata[15: 8];
                if (we && avs_byteenable[2]) m2[w_baddr[AW_W-1:0]] <= avs_writedata[23:16];
                if (we && avs_byteenable[3]) m3[w_baddr[AW_W-1:0]] <= avs_writedata[31:24];
                rd <= {m3[w_raddr], m2[w_raddr], m1[w_raddr], m0[w_raddr]};
            end
            assign w_data[gb*32 +: 32] = rd;
            // synthesis translate_off
            integer wi;
            initial for (wi = 0; wi < W_DEPTH; wi = wi + 1) begin
                m0[wi] = 8'h00; m1[wi] = 8'h00; m2[wi] = 8'h00; m3[wi] = 8'h00;
            end
            // synthesis translate_on
        end
    endgenerate

    // ------------------------------------------------------------ X RAM (write: bus, read: core when BUSY, bus otherwise)
    wire [AW_X-1:0]      x_raddr = busy ? c[AW_X-1:0] : x_baddr[AW_X-1:0];
    wire [LANES*8-1:0]   x_data;

    generate
        for (gb = 0; gb < WB; gb = gb + 1) begin : g_xb
            (* ramstyle = "M10K, no_rw_check" *) reg [7:0] m0 [0:X_DEPTH-1];
            (* ramstyle = "M10K, no_rw_check" *) reg [7:0] m1 [0:X_DEPTH-1];
            (* ramstyle = "M10K, no_rw_check" *) reg [7:0] m2 [0:X_DEPTH-1];
            (* ramstyle = "M10K, no_rw_check" *) reg [7:0] m3 [0:X_DEPTH-1];
            reg [31:0] rd;
            wire we = bus_x_we && (x_bank == gb);
            always @(posedge clk) begin
                if (we && avs_byteenable[0]) m0[x_baddr[AW_X-1:0]] <= avs_writedata[ 7: 0];
                if (we && avs_byteenable[1]) m1[x_baddr[AW_X-1:0]] <= avs_writedata[15: 8];
                if (we && avs_byteenable[2]) m2[x_baddr[AW_X-1:0]] <= avs_writedata[23:16];
                if (we && avs_byteenable[3]) m3[x_baddr[AW_X-1:0]] <= avs_writedata[31:24];
                rd <= {m3[x_raddr], m2[x_raddr], m1[x_raddr], m0[x_raddr]};
            end
            assign x_data[gb*32 +: 32] = rd;
            // synthesis translate_off
            integer xi;
            initial for (xi = 0; xi < X_DEPTH; xi = xi + 1) begin
                m0[xi] = 8'h00; m1[xi] = 8'h00; m2[xi] = 8'h00; m3[xi] = 8'h00;
            end
            // synthesis translate_on
        end
    endgenerate

    // ------------------------------------------------------------ pipeline
    // stage 1: RAM output. stage 2: products (level 0). stages 3..: adder tree.
    reg                    v1, last1;
    reg [LANES-1:0]        mask1;
    reg [LG:0]             vp, lastp;                 // valid/last aligned with tree levels 0..LG
    reg [(LG+1)*LANES*ACCW-1:0] lvl;                  // level l, element i at ((l*LANES)+i)*ACCW

    integer lj, ll, li;
    reg signed [15:0] prod;
    always @(posedge clk) begin
        for (lj = 0; lj < LANES; lj = lj + 1) begin
            prod = $signed(w_data[8*lj +: 8]) * $signed(x_data[8*lj +: 8]);
            lvl[(lj)*ACCW +: ACCW] <= mask1[lj] ? {{(ACCW-16){prod[15]}}, prod} : {ACCW{1'b0}};
        end
        for (ll = 1; ll <= LG; ll = ll + 1)
            for (li = 0; li < (LANES >> ll); li = li + 1)
                lvl[(ll*LANES + li)*ACCW +: ACCW] <=
                      lvl[((ll-1)*LANES + 2*li    )*ACCW +: ACCW]
                    + lvl[((ll-1)*LANES + 2*li + 1)*ACCW +: ACCW];
    end

    wire signed [ACCW-1:0] tree_sum = lvl[(LG*LANES)*ACCW +: ACCW];
    wire signed [31:0]     tree_ext = tree_sum;

    // accumulator and registered Y write
    reg signed [31:0] acc;
    wire signed [31:0] acc_n = acc + tree_ext;
    reg [31:0]        yidx;
    reg               yw_en, yw_fin;
    reg [31:0]        yw_addr;
    reg signed [31:0] yw_data;

    // ------------------------------------------------------------ Y RAM (write: core, read: bus)
    (* ramstyle = "M10K, no_rw_check" *) reg signed [31:0] ymem [0:Y_WORDS-1];
    reg [31:0] y_rd;
    always @(posedge clk) begin
        if (yw_en) ymem[yw_addr[AW_Y-1:0]] <= yw_data;
        y_rd <= ymem[avs_address[AW_Y-1:0]];
    end
    // synthesis translate_off
    integer yi;
    initial for (yi = 0; yi < Y_WORDS; yi = yi + 1) ymem[yi] = 32'sd0;
    // synthesis translate_on

    // ------------------------------------------------------------ control
    integer jm;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            dim_m <= 32'd16;  dim_n <= 32'd16;  gsize <= 32'd0;  cycles <= 32'd0;
            busy <= 1'b0;     done <= 1'b0;     err <= 1'b0;
            cfg_cpr_m1 <= 32'd0; cfg_gch_m1 <= 32'd0; cfg_rows_m1 <= 32'd0;
            cfg_tail <= 32'd1;   cfg_ytotal <= 32'd0;
            issue_active <= 1'b0;
            row <= 32'd0; c <= 32'd0; gcnt <= 32'd0; waddr <= 32'd0;
            v1 <= 1'b0; last1 <= 1'b0; mask1 <= {LANES{1'b0}};
            vp <= {(LG+1){1'b0}}; lastp <= {(LG+1){1'b0}};
            acc <= 32'sd0; yidx <= 32'd0;
            yw_en <= 1'b0; yw_fin <= 1'b0; yw_addr <= 32'd0; yw_data <= 32'sd0;
        end else begin
            // ---- issue stage: one LANES-wide slice per cycle
            v1 <= 1'b0; last1 <= 1'b0; mask1 <= {LANES{1'b0}};
            if (issue_active) begin
                v1    <= 1'b1;
                last1 <= (gcnt == cfg_gch_m1);
                for (jm = 0; jm < LANES; jm = jm + 1)
                    mask1[jm] <= (c != cfg_cpr_m1) || (jm < cfg_tail);
                waddr <= waddr + 32'd1;
                gcnt  <= (gcnt == cfg_gch_m1) ? 32'd0 : (gcnt + 32'd1);
                if (c == cfg_cpr_m1) begin
                    c <= 32'd0;
                    if (row == cfg_rows_m1) issue_active <= 1'b0;
                    else                    row <= row + 32'd1;
                end else begin
                    c <= c + 32'd1;
                end
            end

            // ---- valid/last follow the data through multiplier and tree stages
            vp    <= {vp[LG-1:0],    v1};
            lastp <= {lastp[LG-1:0], last1};

            // ---- accumulate; completed groups go to a registered Y write
            yw_en <= 1'b0; yw_fin <= 1'b0;
            if (vp[LG]) begin
                if (lastp[LG]) begin
                    yw_en   <= 1'b1;
                    yw_addr <= yidx;
                    yw_data <= acc_n;
                    yw_fin  <= ((yidx + 32'd1) == cfg_ytotal);
                    yidx    <= yidx + 32'd1;
                    acc     <= 32'sd0;
                end else begin
                    acc <= acc_n;
                end
            end

            // ---- completion
            if (busy) cycles <= cycles + 32'd1;
            if (yw_en && yw_fin) begin
                busy <= 1'b0;
                done <= 1'b1;
            end

            // ---- bus writes to control registers (bus has priority over the above)
            if (avs_write && ctrl_sel) begin
                case (avs_address[5:0])
                    6'd2: begin  // CTRL
                        if (avs_writedata[1]) begin                  // SOFT_RESET
                            busy <= 1'b0; done <= 1'b0; err <= 1'b0;
                            issue_active <= 1'b0;
                            v1 <= 1'b0; last1 <= 1'b0; mask1 <= {LANES{1'b0}};
                            vp <= {(LG+1){1'b0}}; lastp <= {(LG+1){1'b0}};
                            yw_en <= 1'b0; yw_fin <= 1'b0;
                            acc <= 32'sd0;
                        end else if (avs_writedata[0]) begin         // START
                            if (busy) begin
                                err <= 1'b1;                         // ignored, job continues
                            end else if (cfg_ok) begin
                                busy <= 1'b1; done <= 1'b0; err <= 1'b0;
                                cycles <= 32'd0;
                                cfg_cpr_m1  <= cpr - 32'd1;
                                cfg_gch_m1  <= gch - 32'd1;
                                cfg_rows_m1 <= dim_m - 32'd1;
                                cfg_tail    <= dim_n - ((cpr - 32'd1) << LG);
                                cfg_ytotal  <= y_total;
                                issue_active <= 1'b1;
                                row <= 32'd0; c <= 32'd0; gcnt <= 32'd0; waddr <= 32'd0;
                                yidx <= 32'd0; acc <= 32'sd0;
                            end else begin
                                err <= 1'b1; done <= 1'b0;
                            end
                        end
                    end
                    6'd4: if (!busy) dim_m <= avs_writedata;
                    6'd5: if (!busy) dim_n <= avs_writedata;
                    6'd6: if (!busy) gsize <= avs_writedata;
                    default: ;
                endcase
            end
        end
    end

    // ------------------------------------------------------------ bus reads (latency 1)
    // Selects and register values are captured at the request edge; RAM outputs
    // are already registered, so avs_readdata is a pure mux in the next cycle.
    reg [1:0]  rsel;
    reg [31:0] rreg, xb_r;
    reg        xok_r, yok_r;
    always @(posedge clk) begin
        if (avs_read) begin
            if (ctrl_sel)   rsel <= 2'd0;
            else if (x_sel) rsel <= 2'd1;
            else if (y_sel) rsel <= 2'd2;
            else            rsel <= 2'd3;
            case (avs_address[5:0])
                6'd0:    rreg <= ID_VALUE;
                6'd1:    rreg <= VERSION_VALUE;
                6'd2:    rreg <= 32'd0;
                6'd3:    rreg <= {29'd0, err, done, busy};
                6'd4:    rreg <= dim_m;
                6'd5:    rreg <= dim_n;
                6'd6:    rreg <= gsize;
                6'd7:    rreg <= cycles;
                6'd8:    rreg <= LANES;
                6'd9:    rreg <= {M_MAX_16, N_MAX_16};
                default: rreg <= 32'd0;
            endcase
            xb_r  <= x_bank;
            xok_r <= !busy && (x_baddr < X_DEPTH);
            yok_r <= (avs_address[10:0] < Y_WORDS);
        end
    end

    wire [31:0] x_bus_word = x_data[32*xb_r +: 32];
    assign avs_readdata = (rsel == 2'd0) ? rreg :
                          (rsel == 2'd1) ? (xok_r ? x_bus_word : 32'd0) :
                          (rsel == 2'd2) ? (yok_r ? y_rd       : 32'd0) :
                                           32'd0;

endmodule
