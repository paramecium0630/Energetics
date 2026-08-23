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
        real(dp) :: r_mean ! Mean r
        real(dp) :: r_std ! Standard deviation of r
        real(dp) :: weight_mean ! Mean weight
        real(dp) :: weight_std ! Standard deviation of weight

        ! Noise
        real(dp) :: sigma_mean ! Mean noise strength

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

        real(dp) :: r_mean
        real(dp) :: r_std
        real(dp) :: weight_mean
        real(dp) :: weight_std

        real(dp) :: sigma_mean

        real(dp) :: dt
        real(dp) :: t_relax
        real(dp) :: t_sample
        integer :: lag_steps
        integer :: seed

        ! Namelist definitions
        namelist /network/ N, graph_type, directed, p, network_file

        namelist /dynamics/ r_mean, r_std, weight_mean, weight_std

        namelist /noise/ sigma_mean

        namelist /simulation/ run_simulation, dt, t_relax, t_sample, lag_steps, seed

        ! Default values
        N           = 100
        graph_type  = "ER"
        directed    = .true.
        p           = 0.2_dp
        network_file = "input/weight_matrix.dat"

        r_mean      = 1.0_dp
        r_std       = 0.1_dp
        weight_mean = 1.0_dp
        weight_std  = 0.1_dp

        sigma_mean  = 0.01_dp

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

        read(io_unit, nml=simulation, iostat=io_status)

        if (io_status /= 0) then
            print *, "Error reading &simulation namelist."
            stop
        end if

        close(io_unit)

        ! Store values into param
        param%N           = N
        param%graph_type  = graph_type
        param%directed    = directed
        param%p           = p
        param%network_file = network_file

        param%r_mean      = r_mean
        param%r_std       = r_std ! Default value, can be modified later
        param%weight_mean = weight_mean
        param%weight_std  = weight_std

        param%sigma_mean  = sigma_mean

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

end module parameter_mod