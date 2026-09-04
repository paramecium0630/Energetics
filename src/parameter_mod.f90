module parameter_mod
    use precision_mod
    use random_mod
    implicit none

    type :: SimulationParameters
        character(len=256) :: network_file
        character(len=16) :: graph_type
        ! 最後解析完成的通用資訊
        integer :: N ! no. of nodes
        logical :: directed

        ! ER
        real(dp) :: p

        ! Dynamics
        character(len=256) :: bias_file
        character(len=16) :: bias_mode
        real(dp) :: bias_mean
        real(dp) :: bias_std
        real(dp) :: r_mean ! Mean r
        real(dp) :: r_std ! Standard deviation of r
        real(dp) :: weight_mean ! Mean weight
        real(dp) :: weight_std ! Standard deviation of weight
        character(len=16) :: coupling_type ! Coupling type 

        ! Noise
        real(dp) :: sigma_mean ! Mean noise strength

        ! Theory and verification
        logical :: verify_lyapunov
        integer :: n_weight_shuffles
        integer :: shuffle_seed

        ! Simulation
        logical :: run_simulation
        real(dp) :: dt ! Time step
        real(dp) :: t_relax ! Relaxation time
        real(dp) :: t_sample ! Sampling time
        integer :: lag_steps ! Number of lag steps for correlation calculations

        ! RNG
        integer :: seed ! Seed for random number generator
    end type SimulationParameters 

contains

    subroutine read_parameters(filename, param)
        character(len=*), intent(in) :: filename
        type(SimulationParameters), intent(out) :: param
        ! Read parameters from a file and populate the SimulationParameters type

        integer :: io_unit ! Unit number for file I/O
        integer :: io_status ! Status of I/O operations

        ! Local variables for namelist
        integer :: N
        character(len=16) :: graph_type
        character(len=256) :: network_file
        logical :: directed, run_simulation
        real(dp) :: p

        character(len=256) :: bias_file
        character(len=16) :: bias_mode
        real(dp) :: bias_mean
        real(dp) :: bias_std
        real(dp) :: r_mean
        real(dp) :: r_std
        real(dp) :: weight_mean
        real(dp) :: weight_std
        character(len=16) :: coupling_type

        real(dp) :: sigma_mean

        logical :: verify_lyapunov
        integer :: n_weight_shuffles
        integer :: shuffle_seed

        real(dp) :: dt
        real(dp) :: t_relax
        real(dp) :: t_sample
        integer :: lag_steps
        integer :: seed

        ! Namelist definitions
        namelist /network/ N, graph_type, directed, p, network_file

        namelist /dynamics/ r_mean, r_std, weight_mean, weight_std, &
                            bias_file, bias_mode, bias_mean, bias_std, &
                            coupling_type

        namelist /noise/ sigma_mean

        namelist /theory/ verify_lyapunov, n_weight_shuffles, shuffle_seed

        namelist /simulation/ run_simulation, dt, t_relax, t_sample, lag_steps, seed

        ! Default values
        N           = 100
        graph_type  = "ER"
        directed    = .true.
        p           = 0.2_dp
        network_file = "input/mnistx2/weighted_matrix.dat"
        
        bias_file = "input/mnistx2/bias.dat"
        bias_mode = "AUTO"
        bias_mean = 0.0_dp
        bias_std  = 0.1_dp
        r_mean      = 1.0_dp
        r_std       = 0.1_dp
        weight_mean = 1.0_dp
        weight_std  = 0.1_dp
        coupling_type = "DIFFUSIVE" ! Default coupling type

        sigma_mean  = 0.01_dp

        verify_lyapunov = .true.
        n_weight_shuffles = 0
        shuffle_seed = 1001

        run_simulation = .true.    
        dt          = 0.001_dp
        t_relax     = 100.0_dp
        t_sample    = 1000.0_dp
        lag_steps   = 5

        seed        = 12345

        ! Open input file
        open(newunit=io_unit,              & ! newunit ensures a unique unit number
             file=filename,                &
             status="old",                 &
             action="read",                &
             iostat=io_status)
        
        if (io_status /= 0) then
            print *, "Error: cannot open parameter file."
            print *, "File: ", trim(filename) ! trim removes trailing spaces
            stop
        end if

        ! Read namelists
        read(io_unit, nml=network, iostat=io_status) ! Read the network namelist

        if (io_status /= 0) then
            print *, "Error reading &network namelist."
            stop
        end if

        read(io_unit, nml=dynamics, iostat=io_status)

        if (io_status /= 0) then
            print *, "Error reading &dynamics namelist."
            stop
        end if

        read(io_unit, nml=noise, iostat=io_status)

        if (io_status /= 0) then
            print *, "Error reading &noise namelist."
            stop
        end if

        read(io_unit, nml=theory, iostat=io_status)

        if (io_status /= 0) then
            print *, "Error reading &theory namelist."
            stop
        end if

        read(io_unit, nml=simulation, iostat=io_status)

        if (io_status /= 0) then
            print *, "Error reading &simulation namelist."
            stop
        end if

        close(io_unit)

        ! Validate parameters
        if (n_weight_shuffles < 0) then
            error stop "n_weight_shuffles must be non-negative"
        end if
        if (bias_std < 0.0_dp) then
            error stop "bias_std must be non-negative"
        end if

        ! Store values into param
        param%N           = N
        param%graph_type  = graph_type
        param%directed    = directed
        param%p           = p
        param%network_file = network_file

        param%bias_file   = bias_file
        param%bias_mode   = bias_mode
        param%bias_mean   = bias_mean
        param%bias_std    = bias_std
        param%r_mean      = r_mean
        param%r_std       = r_std ! Default value, can be modified later
        param%weight_mean = weight_mean
        param%weight_std  = weight_std
        param%coupling_type = coupling_type

        param%sigma_mean  = sigma_mean

        param%verify_lyapunov = verify_lyapunov
        param%n_weight_shuffles = n_weight_shuffles
        param%shuffle_seed = shuffle_seed

        param%run_simulation = run_simulation
        param%dt          = dt
        param%t_relax     = t_relax
        param%t_sample    = t_sample
        param%lag_steps   = lag_steps

        param%seed        = seed

    end subroutine read_parameters

    subroutine set_parameters(param, r, noise)
        integer :: i
        type(SimulationParameters), intent(in) :: param ! Input parameters
        real(dp), allocatable, intent(out) :: r(:), noise(:, :) ! Output arrays for r, weight matrix W, and noise
        
        allocate(r(param%N))
        allocate(noise(param%N, param%N))

        r = 0.0_dp
        noise = 0.0_dp

        ! Set r values
        do i = 1, param%N
            ! r(i) = param%r_mean + param%r_std * rand_normal()
            r(i) = param%r_mean
        end do

        ! Set noise values
        do i = 1, param%N
            ! noise(i, i) = param%sigma_mean * (rand_uniform() + 0.5_dp) ! Example: noise on the diagonal
            noise(i, i) = param%sigma_mean
        end do

    end subroutine set_parameters

    subroutine set_random_bias(param, bias)
        type(SimulationParameters), intent(in) :: param
        real(dp), allocatable, intent(out) :: bias(:)

        integer :: i

        if (param%N <= 0) then
            error stop "Number of nodes must be positive"
        end if
        if (param%bias_std < 0.0_dp) then
            error stop "bias_std must be non-negative"
        end if

        allocate(bias(param%N))

        do i = 1, param%N
            bias(i) = param%bias_mean + &
                      param%bias_std * rand_normal()
        end do

    end subroutine set_random_bias

    subroutine initialize_bias(param, bias, bias_layer, n_bias, resolved_mode)
        type(SimulationParameters), intent(in) :: param
        real(dp), allocatable, intent(out) :: bias(:)
        integer, allocatable, intent(out) :: bias_layer(:)
        integer, intent(out) :: n_bias
        character(len=16), intent(out) :: resolved_mode

        resolved_mode = trim(adjustl(param%bias_mode))

        ! AUTO follows the network source:
        ! EXTERNAL reads bias_file; generated networks use random bias.
        if (trim(resolved_mode) == "AUTO") then
            if (trim(adjustl(param%graph_type)) == "EXTERNAL") then
                resolved_mode = "FILE"
            else
                resolved_mode = "RANDOM"
            end if
        end if

        select case (trim(resolved_mode))
        case ("ZERO")
            allocate(bias(param%N), bias_layer(param%N))
            bias = 0.0_dp
            bias_layer = 0
            n_bias = 0

        case ("RANDOM")
            call set_random_bias(param, bias)
            allocate(bias_layer(param%N))
            bias_layer = 0
            n_bias = param%N

        case ("FILE")
            if (len_trim(param%bias_file) == 0) then
                error stop "bias_file must be specified for FILE bias mode"
            end if

            call read_node_bias( &
                trim(param%bias_file), &
                param%N, &
                bias, &
                bias_layer, &
                n_bias)

        case default
            error stop "Unsupported bias mode: " // trim(resolved_mode)
        end select

    end subroutine initialize_bias

    subroutine read_node_bias(filename, n_nodes, bias, bias_layer, n_bias)

        use, intrinsic :: iso_fortran_env, only : iostat_end

        character(len=*), intent(in) :: filename
        integer, intent(in) :: n_nodes ! Number of nodes in the network

        real(dp), allocatable, intent(out) :: bias(:) ! Bias values for each node
        integer, allocatable, intent(out) :: bias_layer(:) ! Layer IDs for each node's bias
        integer, intent(out) :: n_bias ! Number of nodes with bias

        character(len=1024) :: line ! Buffer for reading lines from the file

        integer :: io_unit, io_status, parse_status

        integer :: global_node, layer_id, local_node ! Variables for parsing the bias file

        real(dp) :: bias_value

        logical, allocatable :: bias_is_set(:) ! Logical array to track which nodes have bias set

        if (n_nodes <= 0) then
            error stop "n_nodes must be positive"
        end if

        allocate(bias(n_nodes))
        allocate(bias_layer(n_nodes))
        allocate(bias_is_set(n_nodes))

        ! Nodes without bias remain zero
        bias = 0.0_dp
        bias_layer = 0
        bias_is_set = .false.
        n_bias = 0

        open( &
        newunit=io_unit, &
        file=trim(filename), &
        status="old", &
        action="read", &
        iostat=io_status)

        if (io_status /= 0) then
            error stop "Cannot open bias file"
        end if

        do
        ! Read one complete line
        read(io_unit, '(A)', iostat=io_status) line

        ! End of file
        if (io_status == iostat_end) exit

        ! Actual file-reading error
        if (io_status /= 0) then
            error stop "Error reading bias file"
        end if

        ! Move leading spaces to the end
        line = adjustl(line)

        ! Skip empty lines
        if (len_trim(line) == 0) cycle

        ! Skip comment lines
        if (line(1:1) == "#" .or. line(1:1) == "!") cycle

        ! Parse:
        ! global_node, layer_id, local_node, bias_value
        read(line, *, iostat=parse_status) &
        global_node, layer_id, local_node, bias_value

        if (parse_status /= 0) then
            error stop "Invalid bias-file record"
        end if

        ! Validate global node index
        if (global_node < 1 .or. global_node > n_nodes) then
            error stop "Bias global node is out of range"
        end if

        ! Validate layer index
        if (layer_id <= 0) then
            error stop "Bias layer ID must be positive"
        end if

        ! Validate local node index
        if (local_node <= 0) then
            error stop "Bias local node must be positive"
        end if

        ! Detect duplicate global nodes
        if (bias_is_set(global_node)) then
            error stop "Duplicate global node in bias file"
        end if

        ! Store data using global node index
        bias(global_node) = bias_value
        bias_layer(global_node) = layer_id
        bias_is_set(global_node) = .true.

        n_bias = n_bias + 1
    end do

    close(io_unit, iostat=io_status)

    if (io_status /= 0) then
        error stop "Error closing bias file"
    end if

    if (n_bias == 0) then
        error stop "Bias file contains no bias records"
    end if

    end subroutine read_node_bias

end module parameter_mod
