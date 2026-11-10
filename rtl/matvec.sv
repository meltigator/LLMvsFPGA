// LLMvsFPGA - INT8 matvec accelerator, M0 behavioral version.
//
// Avalon-MM slave (word addressing, fixed read latency 1, no waitrequest).
// Register map and behavior: see docs/architecture.md, section 8.
//
// M0 scope: functionally correct and simulable with Icarus Verilog.
// NOT yet optimized for synthesis:
//   - memories are byte arrays (M1: wide M10K-friendly organization),
//   - the LANES-wide reduction is a single combinational stage
//     (M1: pipelined adder tree, DSP mapping).
//
// Memory layouts (little-endian, byte 0 = bits [7:0] of the word):
//   W window : int8, row-major, index = row*DIM_N + col
//   X window : int8, index = col
//   Y window : int32, index = row*groups + group  (groups = DIM_N/GROUP_SIZE, or 1)

module matvec #(
    parameter LANES   = 16,     // parallel INT8 MACs per cycle
    parameter M_MAX   = 64,     // max rows
    parameter N_MAX   = 1024,   // max columns (multiple of 4)
    parameter Y_WORDS = 2048    // Y window capacity in 32-bit words
) (
    input  wire        clk,
    input  wire        reset_n,

    input  wire [14:0] avs_address,      // WORD address (byte offset >> 2)
    input  wire        avs_read,
    input  wire        avs_write,
    input  wire [31:0] avs_writedata,
    input  wire [3:0]  avs_byteenable,
    output reg  [31:0] avs_readdata      // valid one cycle after avs_read
);

    localparam W_BYTES = M_MAX * N_MAX;

    localparam [31:0] ID_VALUE      = 32'h4C4C4650;  // "LLFP"
    localparam [31:0] VERSION_VALUE = 32'h00000001;

    // ---------------------------------------------------------------- checks
    initial begin
        if (W_BYTES > 65536)       $error("M_MAX*N_MAX exceeds the 64 KB W window");
        if (N_MAX > 4096)          $error("N_MAX exceeds the 4 KB X window");
        if (N_MAX % 4 != 0)        $error("N_MAX must be a multiple of 4");
        if (Y_WORDS > 2048)        $error("Y_WORDS exceeds the 8 KB Y window");
        if (N_MAX >= 65536)        $error("N_MAX must fit in 16 bits");
    end

    // ---------------------------------------------------------------- storage
    reg        [7:0]  wmem [0:W_BYTES-1];
    reg        [7:0]  xmem [0:N_MAX-1];
    reg signed [31:0] ymem [0:Y_WORDS-1];

    // simulation-only initialization, avoids X on never-written locations
    // synthesis translate_off
    integer init_i;
    initial begin
        for (init_i = 0; init_i < W_BYTES; init_i = init_i + 1) wmem[init_i] = 8'h00;
        for (init_i = 0; init_i < N_MAX;   init_i = init_i + 1) xmem[init_i] = 8'h00;
        for (init_i = 0; init_i < Y_WORDS; init_i = init_i + 1) ymem[init_i] = 32'sd0;
    end
    // synthesis translate_on

    // ---------------------------------------------------------------- registers
    reg [31:0] dim_m, dim_n, gsize, cycles;
    reg        busy, done, err;

    // engine state
    reg [31:0]        row;
    reg [31:0]        kpos;
    reg [31:0]        gstart;
    reg [31:0]        yidx;
    reg signed [31:0] acc;

    // ---------------------------------------------------------------- decode
    wire ctrl_sel = (avs_address[14:6] == 9'd0);       // byte 0x0000-0x00FF
    wire x_sel    = (avs_address[14:10] == 5'b00001);  // byte 0x1000-0x1FFF
    wire y_sel    = (avs_address[14:11] == 4'b0001);   // byte 0x2000-0x3FFF
    wire w_sel    = avs_address[14];                   // byte 0x10000-0x1FFFF

    wire [31:0] gs_eff   = (gsize == 32'd0) ? dim_n : gsize;
    wire [31:0] seg_end  = gstart + gs_eff;
    wire [31:0] row_base = row * dim_n;

    // configuration validity, evaluated when START is written
    wire [31:0] groups_per_row = (gsize == 32'd0) ? 32'd1 : (dim_n / gsize);
    wire cfg_ok =
           (dim_m >= 32'd1) && (dim_m <= M_MAX)
        && (dim_n >= 32'd1) && (dim_n <= N_MAX)
        && ((gsize == 32'd0) || ((gsize <= dim_n) && ((dim_n % gsize) == 32'd0)))
        && ((dim_m * groups_per_row) <= Y_WORDS);

    // ---------------------------------------------------------------- main logic
    integer           j, b;
    reg signed [31:0] lane_sum;
    reg signed [31:0] acc_n;
    reg [31:0]        k_n;
    reg [31:0]        widx, xidx;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            dim_m  <= 32'd16;
            dim_n  <= 32'd16;
            gsize  <= 32'd0;
            cycles <= 32'd0;
            busy   <= 1'b0;
            done   <= 1'b0;
            err    <= 1'b0;
            row    <= 32'd0;
            kpos   <= 32'd0;
            gstart <= 32'd0;
            yidx   <= 32'd0;
            acc    <= 32'sd0;
        end else begin

            // ---- compute engine (lower priority than bus: soft reset wins)
            if (busy) begin
                cycles <= cycles + 32'd1;

                lane_sum = 32'sd0;
                for (j = 0; j < LANES; j = j + 1) begin
                    if ((kpos + j) < seg_end)
                        lane_sum = lane_sum
                                 + $signed(wmem[row_base + kpos + j])
                                 * $signed(xmem[kpos + j]);
                end
                acc_n = acc + lane_sum;
                k_n   = kpos + LANES;

                if (k_n >= seg_end) begin
                    // group (or whole row) complete
                    ymem[yidx] <= acc_n;
                    yidx       <= yidx + 32'd1;
                    acc        <= 32'sd0;
                    if (seg_end >= dim_n) begin
                        gstart <= 32'd0;
                        kpos   <= 32'd0;
                        if ((row + 32'd1) >= dim_m) begin
                            busy <= 1'b0;
                            done <= 1'b1;
                        end else begin
                            row <= row + 32'd1;
                        end
                    end else begin
                        gstart <= seg_end;
                        kpos   <= seg_end;
                    end
                end else begin
                    kpos <= k_n;
                    acc  <= acc_n;
                end
            end

            // ---- bus writes
            if (avs_write) begin
                if (ctrl_sel) begin
                    case (avs_address[5:0])
                        6'd2: begin  // CTRL
                            if (avs_writedata[1]) begin          // SOFT_RESET
                                busy <= 1'b0;
                                done <= 1'b0;
                                err  <= 1'b0;
                            end else if (avs_writedata[0]) begin // START
                                if (busy) begin
                                    err <= 1'b1;                 // ignored, job continues
                                end else if (cfg_ok) begin
                                    busy   <= 1'b1;
                                    done   <= 1'b0;
                                    err    <= 1'b0;
                                    cycles <= 32'd0;
                                    row    <= 32'd0;
                                    kpos   <= 32'd0;
                                    gstart <= 32'd0;
                                    yidx   <= 32'd0;
                                    acc    <= 32'sd0;
                                end else begin
                                    err  <= 1'b1;
                                    done <= 1'b0;
                                end
                            end
                        end
                        6'd4: if (!busy) dim_m <= avs_writedata;
                        6'd5: if (!busy) dim_n <= avs_writedata;
                        6'd6: if (!busy) gsize <= avs_writedata;
                        default: ;
                    endcase
                end else if (x_sel && !busy) begin
                    xidx = {20'd0, avs_address[9:0], 2'b00};
                    for (b = 0; b < 4; b = b + 1)
                        if (avs_byteenable[b] && ((xidx + b) < N_MAX))
                            xmem[xidx + b] <= avs_writedata[8*b +: 8];
                end else if (w_sel && !busy) begin
                    widx = {16'd0, avs_address[13:0], 2'b00};
                    for (b = 0; b < 4; b = b + 1)
                        if (avs_byteenable[b] && ((widx + b) < W_BYTES))
                            wmem[widx + b] <= avs_writedata[8*b +: 8];
                end
                // Y window is read-only; writes to it are ignored.
            end
        end
    end

    // ---------------------------------------------------------------- bus reads
    reg [31:0] xr_idx;
    always @(posedge clk) begin
        if (avs_read) begin
            if (ctrl_sel) begin
                case (avs_address[5:0])
                    6'd0:    avs_readdata <= ID_VALUE;
                    6'd1:    avs_readdata <= VERSION_VALUE;
                    6'd2:    avs_readdata <= 32'd0;                       // CTRL reads 0
                    6'd3:    avs_readdata <= {29'd0, err, done, busy};    // STATUS
                    6'd4:    avs_readdata <= dim_m;
                    6'd5:    avs_readdata <= dim_n;
                    6'd6:    avs_readdata <= gsize;
                    6'd7:    avs_readdata <= cycles;
                    6'd8:    avs_readdata <= LANES;                       // CAPS
                    6'd9:    avs_readdata <= {M_MAX[15:0], N_MAX[15:0]};  // LIMITS
                    default: avs_readdata <= 32'd0;
                endcase
            end else if (x_sel) begin
                xr_idx = {20'd0, avs_address[9:0], 2'b00};
                if ((xr_idx + 3) < N_MAX)
                    avs_readdata <= {xmem[xr_idx+3], xmem[xr_idx+2],
                                     xmem[xr_idx+1], xmem[xr_idx]};
                else
                    avs_readdata <= 32'd0;
            end else if (y_sel) begin
                if (avs_address[10:0] < Y_WORDS)
                    avs_readdata <= ymem[avs_address[10:0]];
                else
                    avs_readdata <= 32'd0;
            end else begin
                avs_readdata <= 32'd0;   // W window is write-only
            end
        end
    end

endmodule
