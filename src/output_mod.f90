module output_mod
    use precision_mod
    implicit none

contains

    subroutine write_node_results(filename, r, noise, mean_x, mean_force)
        character(len=*), intent(in) :: filename
        real(dp), intent(in) :: r(:)
        real(dp), intent(in) :: noise(:, :)
        real(dp), intent(in) :: mean_x(:)
        real(dp), intent(in) :: mean_force(:)
        integer :: i, n
        integer :: io_unit

        n = size(r)

        open(newunit=io_unit, file=filename, status='replace', action='write', iostat=i)
        if (i /= 0) error stop "Error opening file for writing: "//filename

        write(io_unit, '(A)') "Node Results"
        write(io_unit, '(A)') "Node Index, r, noise, mean_x, mean_force"
        do i = 1, n
            write(io_unit, '(*(G0,:,","))') i, r(i), noise(i,i), mean_x(i), mean_force(i)
        end do

        close(io_unit)
    end subroutine write_node_results

    subroutine write_edge_results(filename, W, adj_matrix)
        character(len=*), intent(in) :: filename
        logical, intent(in) :: adj_matrix(:, :)
        real(dp), intent(in) :: W(:, :)
        integer :: i, j, n
        integer :: io_unit

        n = size(W, 1)

        open(newunit=io_unit, file=filename, status='replace', action='write', iostat=i)
        if (i /= 0) error stop "Error opening file for writing: "//filename

        write(io_unit, '(A)') "Edge Results"
        write(io_unit, '(A)') "target, source, weight"
        do i = 1, n
            do j = 1, n
                if (.not. adj_matrix(i, j)) cycle
                write(io_unit, '(*(G0,:,","))') i, j, W(i,j)
            end do
        end do

        close(io_unit)
    end subroutine write_edge_results

    subroutine write_jacobian_results(filename, Q)
        character(len=*), intent(in) :: filename
        real(dp), intent(in) :: Q(:, :)
        integer :: i, j, n
        integer :: io_unit

        n = size(Q, 1)

        open(newunit=io_unit, file=filename, status='replace', action='write', iostat=i)
        if (i /= 0) error stop "Error opening file for writing: "//filename

        write(io_unit, '(A)') "Jacobian Results"
        write(io_unit, '(A)') "target, source, Q"
        do i = 1, n
            do j = 1, n
                write(io_unit, '(*(G0,:,","))') i, j, Q(i,j)
            end do
        end do

        close(io_unit)
    end subroutine write_jacobian_results

    subroutine write_correlation_results(filename, K0, Ktau)
        character(len=*), intent(in) :: filename
        real(dp), intent(in) :: K0(:, :), Ktau(:, :)
        integer :: i, j, n
        integer :: io_unit

        n = size(K0, 1)
        
        open(newunit=io_unit, file=filename, status='replace', action='write', iostat=i)
        if (i /= 0) error stop "Error opening file for writing: "//filename

        write(io_unit, '(A)') "Correlation Results"
        write(io_unit, '(A)') "target, source, K0, Ktau"
        do i = 1, n
            do j = 1, n
                write(io_unit, '(*(G0,:,","))') i, j, K0(i,j), Ktau(i,j)
            end do
        end do

        close(io_unit)
    end subroutine write_correlation_results

    subroutine write_alpha(filename, alpha, alpha_sim)
        character(len=*), intent(in) :: filename
        real(dp), intent(in) :: alpha(:, :), alpha_sim(:, :)
        integer :: i, j, n
        integer :: io_unit

        n = size(alpha, 1)
        
        open(newunit=io_unit, file=filename, status='replace', action='write', iostat=i)
        if (i /= 0) error stop "Error opening file for writing: "//filename

        write(io_unit, '(A)') "Correlation Results"
        write(io_unit, '(A)') "target, source, alpha, alpha_sim"
        do i = 1, n
            do j = i+1, n
                write(io_unit, '(*(G0,:,","))') i, j, alpha(i,j), alpha_sim(i,j)
            end do
        end do
        
        close(io_unit)
    end subroutine write_alpha

    subroutine write_energetics_results(filename, heat_rate, work_rate, &
                         internal_rate, entropy_rate)
        character(len=*), intent(in) :: filename
        real(dp), intent(in) :: heat_rate(:), work_rate(:)
        real(dp), intent(in) :: internal_rate(:), entropy_rate(:)
        integer :: i, j, n
        integer :: io_unit

        n = size(heat_rate)

        open(newunit=io_unit, file=filename, status='replace', action='write', iostat=i)
        if (i /= 0) error stop "Error opening file for writing: "//filename

        write(io_unit, '(A)') "Energetics Results"
        write(io_unit, '(A)') "node, heat_rate, entropy_rate, work_rate, internal_rate"

        do i = 1, n
            write(io_unit, '(*(G0,:,","))') i, heat_rate(i), entropy_rate(i), &
            work_rate(i), internal_rate(i)
        end do

        close(io_unit)
    end subroutine write_energetics_results

end module output_mod