module energetics_mod
    use precision_mod
    implicit none

    type, public :: EnergeticsState
        integer :: n_step = 0
        real(dp) :: elapsed_time = 0.0_dp

        real(dp), allocatable :: S(:,:)
        real(dp), allocatable :: A(:,:)
        real(dp), allocatable :: sigma_diag(:)

        real(dp), allocatable :: sum_heat(:)
        ! real(dp), allocatable :: sum_entropy(:)
        real(dp), allocatable :: sum_work(:)
        real(dp), allocatable :: sum_internal(:)

        ! Reusable workspace for each sampling step
        real(dp), allocatable :: dy(:)
        real(dp), allocatable :: y_mid(:)
        real(dp), allocatable :: force_c(:)
        real(dp), allocatable :: force_nc(:)

    end type EnergeticsState

    public :: initialize_energetics
    public :: update_energetics_linear
    public :: finalize_energetics

contains

    subroutine initialize_energetics(energy, Q, noise)
        type(EnergeticsState), intent(out) :: energy
        integer :: n, i
        real(dp), intent(in) :: Q(:,:), noise(:,:)

        n = size(Q, 1)

        allocate(energy%S(n,n), energy%A(n,n), energy%sigma_diag(n))

        allocate(energy%sum_heat(n))
        allocate(energy%sum_work(n))
        allocate(energy%sum_internal(n))

        allocate(energy%dy(n))
        allocate(energy%y_mid(n))
        allocate(energy%force_c(n))
        allocate(energy%force_nc(n))

        if (size(Q,2) /= n) error stop "Q must be square"

        if (size(noise,1) /= n .or. size(noise,2) /= n) then
            error stop "Q and noise size mismatch"
        end if

        ! decompose force
        energy%S = 0.5_dp * (Q + transpose(Q))
        energy%A = 0.5_dp * (Q - transpose(Q))

        ! noise diagonal
        do i = 1, n
            if (noise(i,i) <= 0.0_dp) then
                error stop "Noise diagonal must be positive"
            end if

            energy%sigma_diag(i) = noise(i,i)
        end do

        energy%sum_heat     = 0.0_dp
        energy%sum_work     = 0.0_dp
        energy%sum_internal = 0.0_dp
        energy%dy           = 0.0_dp
        energy%y_mid        = 0.0_dp
        energy%force_c      = 0.0_dp
        energy%force_nc     = 0.0_dp
        energy%n_step       = 0
        energy%elapsed_time = 0.0_dp

    end subroutine initialize_energetics

    subroutine update_energetics_linear(energy, y_old, y_new, dt)
        type(EnergeticsState), intent(inout) :: energy
        integer :: n, i
        real(dp), intent(in) :: y_old(:), y_new(:)
        real(dp), intent(in) :: dt

        n = size(y_old)

        if (size(y_new) /= n) then
            error stop "y_old and y_new size mismatch"
        end if
        if (.not. allocated(energy%dy)) then
            error stop "Energetics workspace is not initialized"
        end if
        if (size(energy%dy) /= n) then
            error stop "Energetics workspace size mismatch"
        end if
        if (dt <= 0.0_dp) error stop "dt must be positive"

        ! dx(t)
        energy%dy    = y_new - y_old
        energy%y_mid = 0.5_dp * (y_old + y_new)

        ! conservative and nonconservative force
        energy%force_c  = matmul(energy%S, energy%y_mid)
        energy%force_nc = matmul(energy%A, energy%y_mid)

        do i = 1, n
            energy%sum_heat(i) = energy%sum_heat(i) - &
                (energy%force_c(i) + energy%force_nc(i)) * energy%dy(i)
            energy%sum_work(i) = energy%sum_work(i) - &
                energy%force_nc(i) * energy%dy(i)
            energy%sum_internal(i) = energy%sum_internal(i) - &
                energy%force_c(i) * energy%dy(i)
        end do

        energy%n_step       = energy%n_step + 1
        energy%elapsed_time = energy%elapsed_time + dt

    end subroutine update_energetics_linear 

    subroutine finalize_energetics(energy, heat_rate, work_rate, &
                               internal_rate, entropy_rate)
        type(EnergeticsState), intent(inout) :: energy

        real(dp), allocatable, intent(out) :: heat_rate(:)
        real(dp), allocatable, intent(out) :: work_rate(:)
        real(dp), allocatable, intent(out) :: internal_rate(:)
        real(dp), allocatable, intent(out) :: entropy_rate(:)

        heat_rate     = energy%sum_heat     / energy%elapsed_time
        work_rate     = energy%sum_work     / energy%elapsed_time
        internal_rate = energy%sum_internal / energy%elapsed_time

        entropy_rate = -2.0_dp * heat_rate / energy%sigma_diag

        if (allocated(energy%S)) deallocate(energy%S)
        if (allocated(energy%A)) deallocate(energy%A)
        if (allocated(energy%sigma_diag)) deallocate(energy%sigma_diag)
        if (allocated(energy%sum_heat)) deallocate(energy%sum_heat)
        if (allocated(energy%sum_work)) deallocate(energy%sum_work)
        if (allocated(energy%sum_internal)) deallocate(energy%sum_internal)
        if (allocated(energy%dy)) deallocate(energy%dy)
        if (allocated(energy%y_mid)) deallocate(energy%y_mid)
        if (allocated(energy%force_c)) deallocate(energy%force_c)
        if (allocated(energy%force_nc)) deallocate(energy%force_nc)

        energy%n_step = 0
        energy%elapsed_time = 0.0_dp

    end subroutine finalize_energetics

end module energetics_mod
