module output_mod
    use precision_mod
    implicit none

contains

    subroutine write_node_results( &
        filename, r, noise, fixpoint, bias, node_layer)
        character(len=*), intent(in) :: filename
        real(dp), intent(in) :: r(:)
        real(dp), intent(in) :: noise(:, :)
        real(dp), intent(in) :: fixpoint(:)
        real(dp), intent(in) :: bias(:)
        integer, intent(in), optional :: node_layer(:)
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
        if (present(node_layer)) then
            if (size(node_layer) /= n) then
                error stop "r and node_layer size mismatch"
            end if
            if (any(node_layer <= 0)) then
                error stop "FCNN layer indices must be positive"
            end if
        end if

        open(newunit=io_unit, file=filename, status='replace', action='write', iostat=i)
        if (i /= 0) error stop "Error opening file for writing: "//filename

        write(io_unit, '(A)') "Node Results"
        write(io_unit, '(A)') &
            "Node Index, layer, r, noise, fixpoint, bias"
        do i = 1, n
            if (present(node_layer)) then
                ! FCNN layer indices are one-based: input layer is layer 1.
                write(io_unit, '(*(G0,:,","))') &
                    i, node_layer(i), r(i), noise(i,i), &
                    fixpoint(i), bias(i)
            else
                ! Layer 0 means that no FCNN layer assignment is available.
                write(io_unit, '(*(G0,:,","))') &
                    i, 0, r(i), noise(i,i), fixpoint(i), bias(i)
            end if
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
                         internal_rate, entropy_rate, node_layer)
        ! Canonical node output: one header, one row per node. Layer=0 for non-FCNN.
        character(len=*), intent(in) :: filename
        real(dp), intent(in) :: heat_rate(:), work_rate(:)
        real(dp), intent(in) :: internal_rate(:), entropy_rate(:)
        integer, intent(in) :: node_layer(:)
        integer :: node, n, io_unit, io_status
        n = size(heat_rate)
        if (n <= 0 .or. size(node_layer) /= n .or. size(work_rate) /= n .or. &
            size(internal_rate) /= n .or. size(entropy_rate) /= n) then
            error stop "Node energetics size mismatch"
        end if
        if (any(node_layer < 0)) error stop "Negative layer ID"
        open(newunit=io_unit, file=filename, status="replace", action="write", iostat=io_status)
        if (io_status /= 0) error stop "Cannot open node energetics output"
        write(io_unit, '(A)') "Node,Layer,HR,EPR,WR,UR"
        do node = 1, n
            write(io_unit, '(*(G0,:,","))') node, node_layer(node), &
                heat_rate(node), entropy_rate(node), work_rate(node), internal_rate(node)
        end do
        close(io_unit)
    end subroutine write_energetics_results

    subroutine write_node_layers(filename, node_layer)
        character(len=*), intent(in) :: filename
        integer, intent(in) :: node_layer(:)
        integer :: node, io_unit, io_status
        if (size(node_layer) == 0 .or. any(node_layer <= 0)) error stop "Invalid FCNN layers"
        open(newunit=io_unit, file=filename, status="replace", action="write", iostat=io_status)
        if (io_status /= 0) error stop "Cannot open node-layer mapping"
        write(io_unit, '(A)') "Node,Layer"
        do node = 1, size(node_layer)
            write(io_unit, '(*(G0,:,","))') node, node_layer(node)
        end do
        close(io_unit)
    end subroutine write_node_layers

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
        if (any(node_layer < 1)) then
            error stop "Layer IDs must be positive"
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


    subroutine write_layer_energetics_results( &
        filename, node_count, layer_heat, layer_work, layer_internal, layer_entropy)
        ! 每層一列；同時輸出層總值及每節點平均，避免兩種量混用。
        character(len=*), intent(in) :: filename
        integer, intent(in) :: node_count(:)
        real(dp), intent(in) :: layer_heat(:), layer_work(:)
        real(dp), intent(in) :: layer_internal(:), layer_entropy(:)
        integer :: n_layers, layer, io_unit, io_status
        real(dp) :: count_real

        n_layers = size(node_count)
        if (n_layers <= 0) error stop "Layer output requires at least one layer"
        if (any(node_count <= 0)) error stop "Layer node counts must be positive"
        if (size(layer_heat) /= n_layers .or. size(layer_work) /= n_layers .or. &
            size(layer_internal) /= n_layers .or. size(layer_entropy) /= n_layers) then
            error stop "Layer counts and energetic rates size mismatch"
        end if
        open(newunit=io_unit, file=filename, status="replace", &
             action="write", iostat=io_status)
        if (io_status /= 0) error stop "Error opening file for writing: " // filename
        ! 沿用既有輸出格式：第一列標題，第二列 CSV 欄名。
        write(io_unit, '(A)') "Energetics by Layer"
        write(io_unit, '(A)') &
            "layer,node_count,heat_total,entropy_total,work_total,internal_total," // &
            "heat_per_node,entropy_per_node,work_per_node,internal_per_node"
        do layer = 1, n_layers
            count_real = real(node_count(layer), dp)
            write(io_unit, '(*(G0,:,","))') layer, node_count(layer), &
                layer_heat(layer), layer_entropy(layer), &
                layer_work(layer), layer_internal(layer), &
                layer_heat(layer) / count_real, layer_entropy(layer) / count_real, &
                layer_work(layer) / count_real, layer_internal(layer) / count_real
        end do
        close(io_unit)
    end subroutine write_layer_energetics_results

end module output_mod
