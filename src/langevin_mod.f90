module langevin_mod
    use precision_mod
    use parameter_mod
    use random_mod
    implicit none

contains

    subroutine construct_Q(r, W, Q)
        ! Construct the Q matrix for the simulation
        ! Diffusive coupling between target and source states
        ! The Q matrix is defined as:
        ! Q_ij = W_ij for i ≠ j
        ! Q_ii = -r_i - sum_j W_ij
        integer :: i, n
        real(dp), intent(in) :: r(:)
        real(dp), intent(in) :: W(:,:)
        real(dp), allocatable, intent(out) :: Q(:,:)

        n = size(r)

        if (size(W, 1) /= n .or. size(W, 2) /= n) then
            error stop "Dimension mismatch between r and W" ! 
        end if

        allocate(Q(n, n))

        ! Initialize Q with W and set diagonal elements
        Q = W ! Q_ij = W_ij for i ≠ j
        do i = 1, n
            Q(i, i) = -r(i) - sum(W(i, :)) + W(i, i) ! Q_ii = -r_i - sum_j W_ij
            ! Q(i, i) = -r(i)
        end do
    end subroutine construct_Q

    subroutine initialize_state(n, x, delta_x, force)
        integer, intent(in) :: n
        real(dp), allocatable, intent(out) :: x(:), delta_x(:), force(:)
        ! Initialize the state vector x with random values

        if (n <= 0) error stop "State size must be positive"

        allocate(x(n), delta_x(n), force(n))
        
        x = 1.0_dp
        ! x = 0.0_dp
        delta_x = 0.0_dp
        force = 0.0_dp

    end subroutine initialize_state

    subroutine compute_force(x, r, W, bias, coupling_type, force)
        ! Compute the deterministic force based on the current state x
        integer :: n, i, j
        real(dp) :: source_value
        real(dp), intent(in) :: x(:)
        real(dp), intent(in) :: r(:)
        real(dp), intent(in) :: W(:,:)
        real(dp), intent(in) :: bias(:)
        character(len=16), intent(in) :: coupling_type
        real(dp), intent(out) :: force(:)

        n = size(x)

        if (size(r) /= n) error stop "x and r size mismatch"
        if (size(force) /= n) error stop "x and force size mismatch"
        if (size(W, 1) /= n .or. size(W, 2) /= n) then
            error stop "x and W size mismatch"
        end if
        if (size(bias) /= n) then
            error stop "x and bias size mismatch"
        end if

        force = -r * x + bias ! force = -r * x + bias

        select case (trim(adjustl(coupling_type)))

        case ("DIFFUSIVE")

            ! sum_j W(i,j) * (x(j) - x(i))
            do j = 1, n
                do i = 1, n
                    force(i) = force(i) + &
                    W(i, j) * (x(j) - x(i))
                end do
            end do

        case ("TANH")

            ! sum_j W(i,j) * tanh(x(j))
            do j = 1, n
                source_value = tanh(x(j))
                do i = 1, n
                    force(i) = force(i) + &
                    W(i, j) * source_value
                end do
            end do

        case default

            error stop "Unsupported coupling type: " // &
            trim(coupling_type)

        end select

    end subroutine compute_force

    subroutine langevin_step(x, force, noise, dt)
        ! Perform a single Langevin step
        integer :: n, i, j
        real(dp), intent(inout) :: x(:)
        real(dp), intent(in) :: force(:)
        real(dp), intent(in) :: noise(:,:)
        real(dp), intent(in) :: dt

        n = size(x)

        if (size(noise, 1) /= n .or. size(noise, 2) /= n) then
            error stop "x and noise size mismatch"
        end if
        if (dt <= 0.0_dp) error stop "dt must be positive"

        ! Update the state vector x using the Langevin equation
        do i = 1, n
            if (noise(i, i) < 0.0_dp) then
                error stop "Negative noise covariance"
            end if
            x(i) = x(i) + dt * force(i) + sqrt(noise(i,i)*dt) * rand_normal()
        end do        

    end subroutine langevin_step

end module langevin_mod
