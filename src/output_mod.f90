module output_mod
    use precision_mod
    implicit none

contains

    subroutine write_node_results(filename, r, noise, fixpoint, bias)
        character(len=*), intent(in) :: filename
        real(dp), intent(in) :: r(:)
        real(dp), intent(in) :: noise(:, :)
        real(dp), intent(in) :: fixpoint(:)
        real(dp), intent(in) :: bias(:)
        integer :: i, n
        integer :: io_unit

        n = size(r)

        if (size(noise, 1) /= n .or. size(noise, 2) /= n) then
            error stop "r and noise size mismatch"
        end if
        if (size(fixpoint) /= n) then
            error stop "r and fixpoint size mismatch"
        end if
        if (size(bias) /= n) then
            error stop "r and bias size mismatch"
        end if

        open(newunit=io_unit, file=filename, status='replace', action='write', iostat=i)
        if (i /= 0) error stop "Error opening file for writing: "//filename

        write(io_unit, '(A)') "Node Results"
        write(io_unit, '(A)') "Node Index, r, noise, fixpoint, bias"
        do i = 1, n
            write(io_unit, '(*(G0,:,","))') &
                i, r(i), noise(i,i), fixpoint(i), bias(i)
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

    subroutine write_mean_results(filename, mean_x, mean_f)
        character(len=*), intent(in) :: filename
        real(dp), intent(in) :: mean_x(:), mean_f(:)
        integer :: i, n
        integer :: io_unit

        n = size(mean_x)

        open(newunit=io_unit, file=filename, status='replace', action='write', iostat=i)
        if (i /= 0) error stop "Error opening file for writing: "//filename

        write(io_unit, '(A)') "Mean Results"
        write(io_unit, '(A)') "Node, <x>, <F>"

        do i = 1, n
            write(io_unit, '(*(G0,:,","))') i, mean_x(i), mean_f(i)
        end do

        close(io_unit)
    end subroutine write_mean_results

    subroutine write_correlation_results(filename, K0, Ktau, K0_theory)
        character(len=*), intent(in) :: filename
        real(dp), intent(in) :: K0(:, :), Ktau(:, :), K0_theory(:, :)
        integer :: i, j, n
        integer :: io_unit

        n = size(K0, 1)

        if (size(K0, 2) /= n) error stop "K0 must be square"
        if (size(Ktau, 1) /= n .or. size(Ktau, 2) /= n) then
            error stop "K0 and Ktau size mismatch"
        end if
        if (size(K0_theory, 1) /= n .or. size(K0_theory, 2) /= n) then
            error stop "K0 and K0_theory size mismatch"
        end if
        
        open(newunit=io_unit, file=filename, status='replace', action='write', iostat=i)
        if (i /= 0) error stop "Error opening file for writing: "//filename

        write(io_unit, '(A)') "Correlation Results"
        write(io_unit, '(A)') "target, source, K0, Ktau, K0_theory"
        do i = 1, n
            do j = 1, n
                write(io_unit, '(*(G0,:,","))') &
                    i, j, K0(i,j), Ktau(i,j), K0_theory(i,j)
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

    subroutine write_energetics_by_node_and_layer( &
        filename, node_layer, heat_rate, work_rate, &
        internal_rate, entropy_rate)
        character(len=*), intent(in) :: filename
        integer, intent(in) :: node_layer(:)
        real(dp), intent(in) :: heat_rate(:), work_rate(:)
        real(dp), intent(in) :: internal_rate(:), entropy_rate(:)
        integer :: n, node, io_unit, io_status

        n = size(node_layer)

        if (n <= 0) error stop "Layer output requires at least one node"
        if (any(node_layer < 0)) then
            error stop "Layer IDs must be non-negative"
        end if
        if (size(heat_rate) /= n .or. size(work_rate) /= n .or. &
            size(internal_rate) /= n .or. size(entropy_rate) /= n) then
            error stop "Layer IDs and energetic rates size mismatch"
        end if

        open(newunit=io_unit, file=filename, status="replace", &
             action="write", iostat=io_status)
        if (io_status /= 0) then
            error stop "Error opening file for writing: " // filename
        end if

        write(io_unit, '(A)') &
            "Theoretical Energetics by Node and Layer"
        write(io_unit, '(A)') &
            "layer,node,heat_rate,entropy_rate," // &
            "work_rate,internal_rate"

        do node = 1, n
            write(io_unit, '(*(G0,:,","))') &
                node_layer(node), node, &
                heat_rate(node), entropy_rate(node), &
                work_rate(node), internal_rate(node)
        end do

        close(io_unit)

    end subroutine write_energetics_by_node_and_layer

end module output_mod
