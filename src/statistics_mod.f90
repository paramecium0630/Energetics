module statistics_mod
    use precision_mod
    implicit none

    type :: StatisticsState
    integer :: n_sample
    integer :: lag_steps
    integer :: n_lag_pairs

    real(dp), allocatable :: sum_y(:)
    real(dp), allocatable :: sum_force(:)
    real(dp), allocatable :: sum_yy(:,:)
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

        allocate(stat%sum_y(n))
        allocate(stat%sum_force(n))
        allocate(stat%sum_yy(n,n))
        allocate(stat%sum_lag(n,n))
        allocate(stat%history(n, lag_steps))

        stat%sum_y = 0.0_dp ! y = x - 1
        stat%sum_force = 0.0_dp
        stat%sum_yy = 0.0_dp ! yy = y * y^T
        stat%sum_lag = 0.0_dp ! lagged correlations
        stat%history = 0.0_dp ! history for lagged correlations
    end subroutine initialize_statistics

    subroutine update_statistics(stat, y, force)
        type(StatisticsState), intent(inout) :: stat
        real(dp), intent(in) :: y(:)
        real(dp), intent(in) :: force(:)

        integer :: n, i

        n = size(y)

        if (size(force) /= n) error stop "force size mismatch"
        if (size(stat%sum_y) /= n) error stop "statistics size mismatch"

        ! Update sums
        stat%sum_y = stat%sum_y + y ! y = x - 1
        stat%sum_force = stat%sum_force + force 
        do i = 1, n
            stat%sum_yy(i, :) = stat%sum_yy(i, :) + y(i) * y ! yy = y * y^T
        end do

        ! Update lagged sums
        if (stat%n_sample >= stat%lag_steps) then            
            do i = 1, n
                stat%sum_lag(i, :) = stat%sum_lag(i, :) + y(i) * stat%history(:, 1) ! lagged correlation
            end do
            stat%n_lag_pairs = stat%n_lag_pairs + 1 ! Increment lagged pair count
        end if

        ! Update history for lagged correlations
        if (stat%lag_steps > 1) then
            stat%history(:, 1:stat%lag_steps-1) = stat%history(:, 2:stat%lag_steps) ! Shift history to the left
        end if
        stat%history(:, stat%lag_steps) = y

        ! Increment sample count
        stat%n_sample = stat%n_sample + 1
    end subroutine update_statistics

    subroutine finalize_statistics(stat, fixpoint, mean_x, mean_force, K0, Ktau)
        type(StatisticsState), intent(inout) :: stat
        integer :: n
        real(dp), intent(in) :: fixpoint(:)
        real(dp) :: inv_samples, inv_lag_pairs
        real(dp), allocatable, intent(out) :: mean_x(:), mean_force(:), K0(:,:), Ktau(:,:)
        real(dp), allocatable :: mean_y(:)

        if (stat%n_sample <= 0) then
            error stop "No samples collected"
        end if

        if (stat%n_lag_pairs <= 0) then
            error stop "No valid lag pairs collected"
        end if

        n = size(stat%sum_y)

        allocate(mean_x(n), mean_force(n), K0(n,n), Ktau(n,n), mean_y(n))

        inv_samples = 1.0_dp / real(stat%n_sample, dp) ! Inverse of sample size
        inv_lag_pairs = 1.0_dp / real(stat%n_lag_pairs, dp) ! Inverse of lag pairs

        mean_y = stat%sum_y * inv_samples
        mean_x = fixpoint + mean_y
        mean_force = stat%sum_force * inv_samples

        K0 = stat%sum_yy * inv_samples
        Ktau = stat%sum_lag * inv_lag_pairs

        ! Deallocate arrays
        if (allocated(stat%sum_y)) deallocate(stat%sum_y)
        if (allocated(stat%sum_force)) deallocate(stat%sum_force)
        if (allocated(stat%sum_yy)) deallocate(stat%sum_yy)
        if (allocated(stat%sum_lag)) deallocate(stat%sum_lag)
        if (allocated(stat%history)) deallocate(stat%history)

        ! Reset sample count
        stat%n_sample = 0
        stat%n_lag_pairs = 0

    end subroutine finalize_statistics

end module statistics_mod