module statistics_mod
    use precision_mod
    implicit none

    type :: StatisticsState
    integer :: n_sample
    integer :: lag_steps
    integer :: n_lag_pairs

    real(dp), allocatable :: sum_delta_x(:)
    real(dp), allocatable :: sum_force(:)
    real(dp), allocatable :: sum_delta_xx(:,:)
    real(dp), allocatable :: sum_lag(:,:)

    real(dp), allocatable :: history(:,:)
end type StatisticsState

contains
    ! Initialize the statistics state
    subroutine initialize_statistics(stat, lag_steps, n)
        type(StatisticsState), intent(out) :: stat
        integer, intent(in) :: lag_steps, n

        if (n <= 0) error stop "Number of nodes must be positive"
        if (lag_steps <= 0) error stop "lag_steps must be positive"

        stat%n_sample = 0
        stat%lag_steps = lag_steps
        stat%n_lag_pairs = 0

        allocate(stat%sum_delta_x(n))
        allocate(stat%sum_force(n))
        allocate(stat%sum_delta_xx(n,n))
        allocate(stat%sum_lag(n,n))
        allocate(stat%history(n, lag_steps))

        stat%sum_delta_x = 0.0_dp
        stat%sum_force = 0.0_dp
        stat%sum_delta_xx = 0.0_dp
        stat%sum_lag = 0.0_dp ! lagged correlations
        stat%history = 0.0_dp ! history for lagged correlations
    end subroutine initialize_statistics

    subroutine update_statistics(stat, delta_x, force)
        type(StatisticsState), intent(inout) :: stat
        real(dp), intent(in) :: delta_x(:)
        real(dp), intent(in) :: force(:)

        integer :: n
        external :: dger

        n = size(delta_x)

        if (size(force) /= n) error stop "force size mismatch"
        if (size(stat%sum_delta_x) /= n) error stop "statistics size mismatch"

        ! Update sums
        stat%sum_delta_x = stat%sum_delta_x + delta_x
        stat%sum_force = stat%sum_force + force 

        ! sum_delta_xx += delta_x(t) * delta_x(t)^T
        call dger(n, n, 1.0_dp, delta_x, 1, delta_x, 1, &
                  stat%sum_delta_xx, n)

        ! Update lagged sums
        if (stat%n_sample >= stat%lag_steps) then            
            ! sum_lag += delta_x(t) * delta_x(t-tau)^T
            call dger(n, n, 1.0_dp, delta_x, 1, &
                      stat%history(:, 1), 1, stat%sum_lag, n)

            stat%n_lag_pairs = stat%n_lag_pairs + 1 ! Increment lagged pair count
        end if

        ! Update history for lagged correlations
        if (stat%lag_steps > 1) then
            stat%history(:, 1:stat%lag_steps-1) = stat%history(:, 2:stat%lag_steps) ! Shift history to the left
        end if
        stat%history(:, stat%lag_steps) = delta_x

        ! Increment sample count
        stat%n_sample = stat%n_sample + 1
    end subroutine update_statistics

    subroutine finalize_statistics(stat, fixpoint, mean_x, mean_force, K0, Ktau)
        type(StatisticsState), intent(inout) :: stat
        integer :: n
        real(dp), intent(in) :: fixpoint(:)
        real(dp) :: inv_samples, inv_lag_pairs
        real(dp), allocatable, intent(out) :: mean_x(:), mean_force(:), K0(:,:), Ktau(:,:)
        real(dp), allocatable :: mean_delta_x(:)

        if (stat%n_sample <= 0) then
            error stop "No samples collected"
        end if

        if (stat%n_lag_pairs <= 0) then
            error stop "No valid lag pairs collected"
        end if

        n = size(stat%sum_delta_x)

        allocate(mean_x(n), mean_force(n), K0(n,n), Ktau(n,n), &
                 mean_delta_x(n))

        inv_samples = 1.0_dp / real(stat%n_sample, dp) ! Inverse of sample size
        inv_lag_pairs = 1.0_dp / real(stat%n_lag_pairs, dp) ! Inverse of lag pairs

        mean_delta_x = stat%sum_delta_x * inv_samples
        mean_x = fixpoint + mean_delta_x
        mean_force = stat%sum_force * inv_samples

        K0 = stat%sum_delta_xx * inv_samples
        Ktau = stat%sum_lag * inv_lag_pairs

        ! Deallocate arrays
        if (allocated(stat%sum_delta_x)) deallocate(stat%sum_delta_x)
        if (allocated(stat%sum_force)) deallocate(stat%sum_force)
        if (allocated(stat%sum_delta_xx)) deallocate(stat%sum_delta_xx)
        if (allocated(stat%sum_lag)) deallocate(stat%sum_lag)
        if (allocated(stat%history)) deallocate(stat%history)

        ! Reset sample count
        stat%n_sample = 0
        stat%n_lag_pairs = 0

    end subroutine finalize_statistics

end module statistics_mod
