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
        energy%n_step       = 0
        energy%elapsed_time = 0.0_dp

    end subroutine initialize_energetics

    subroutine update_energetics_linear(energy, y_old, y_new, dt)
        type(EnergeticsState), intent(inout) :: energy
        integer :: n
        real(dp), intent(in) :: y_old(:), y_new(:)
        real(dp), allocatable :: dy(:), y_mid(:), force_c(:), force_nc(:), force(:)
        real(dp), allocatable :: dheat(:), dwork(:), dinternal(:)
        real(dp), intent(in) :: dt

        n = size(y_old, 1)

        allocate(dy(n), y_mid(n), force_c(n), force_nc(n), force(n))
        allocate(dheat(n), dwork(n), dinternal(n))

        ! dx(t)
        dy    = y_new - y_old
        y_mid = 0.5_dp * (y_old + y_new)

        ! conservative and nonconservative force
        force_c  = matmul(energy%S, y_mid)
        force_nc = matmul(energy%A, y_mid)
        force    = force_c + force_nc

        dheat     = -force    * dy
        dwork     = -force_nc * dy
        dinternal = -force_c  * dy

        energy%sum_heat     = energy%sum_heat     + dheat
        energy%sum_work     = energy%sum_work     + dwork
        energy%sum_internal = energy%sum_internal + dinternal

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

    end subroutine finalize_energetics

end module energetics_mod