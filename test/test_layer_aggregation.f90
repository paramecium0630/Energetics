program test_layer_aggregation
    use precision_mod
    use energetics_mod, only : aggregate_layer_energetics
    use output_mod, only : write_layer_energetics_results, write_energetics_results
    implicit none
    real(dp), parameter :: tol = 1.0e-12_dp
    integer :: labels(5), unit, status, ell, read_layer, read_count
    integer, allocatable :: counts(:)
    real(dp) :: heat(5), work(5), internal(5), entropy(5), sigma(5)
    real(dp), allocatable :: lh(:), lw(:), lu(:), ls(:)
    real(dp) :: totals(4), averages(4), expected(4)
    character(len=512) :: line
    character(len=*), parameter :: filename = "test_layer_aggregation.csv"

    ! 刻意交錯節點分層，並令同層節點噪音不同。
    labels = [2, 1, 3, 1, 2]
    heat = [-1.0_dp, -2.0_dp, -3.0_dp, -4.0_dp, -5.0_dp]
    work = [-0.5_dp, -1.5_dp, -1.0_dp, -2.0_dp, -4.0_dp]
    internal = heat - work
    sigma = [1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp]
    entropy = -2.0_dp * heat / sigma
    call aggregate_layer_energetics( &
        labels, heat, work, internal, entropy, counts, lh, lw, lu, ls)
    if (any(counts /= [2, 2, 1])) error stop "Incorrect layer counts"
    if (maxval(abs(lh - [-6.0_dp, -6.0_dp, -3.0_dp])) > tol) error stop "Layer heat"
    if (maxval(abs(lw - [-3.5_dp, -4.5_dp, -1.0_dp])) > tol) error stop "Layer work"
    if (maxval(abs(lu - [-2.5_dp, -1.5_dp, -2.0_dp])) > tol) error stop "Layer internal"
    if (maxval(abs(ls - [4.0_dp, 4.0_dp, 2.0_dp])) > tol) error stop "Layer entropy"
    if (maxval(abs(lh - lw - lu)) > tol) error stop "Layer energy balance"
    if (abs(sum(lh)-sum(heat)) > tol .or. abs(sum(lw)-sum(work)) > tol .or. &
        abs(sum(lu)-sum(internal)) > tol .or. abs(sum(ls)-sum(entropy)) > tol) then
        error stop "Layer totals do not preserve network totals"
    end if

    call write_layer_energetics_results(filename, counts, lh, lw, lu, ls)
    open(newunit=unit, file=filename, status="old", action="read")
    read(unit, '(A)') line
    read(unit, '(A)') line
    if (trim(line) /= &
        "layer,node_count,heat_total,entropy_total,work_total,internal_total," // &
        "heat_per_node,entropy_per_node,work_per_node,internal_per_node") then
        error stop "Incorrect CSV header"
    end if
    do ell = 1, 3
        read(unit, *, iostat=status) read_layer, read_count, totals, averages
        if (status /= 0) error stop "Missing layer CSV row"
        if (read_layer /= ell .or. read_count /= counts(ell)) error stop "CSV layer ID/count"
        expected = [lh(ell), ls(ell), lw(ell), lu(ell)]
        if (maxval(abs(totals - expected)) > tol) error stop "CSV totals"
        if (maxval(abs(averages - expected / real(counts(ell), dp))) > tol) error stop "CSV averages"
    end do
    read(unit, '(A)', iostat=status) line
    if (status >= 0) error stop "Unexpected extra CSV row"
    close(unit, status="delete")

    ! Canonical node table has exactly six columns and no descriptive title.
    call write_energetics_results(filename, heat, work, internal, entropy, labels)
    open(newunit=unit, file=filename, status="old", action="read")
    read(unit, '(A)') line
    if (trim(line) /= "Node,Layer,HR,EPR,WR,UR") error stop "Node CSV header"
    do ell = 1, 5
        read(unit, *, iostat=status) read_count, read_layer, totals
        if (status /= 0) error stop "Missing node rate row"
        if (read_count /= ell .or. read_layer /= labels(ell)) error stop "Node CSV identifiers"
        expected = [heat(ell), entropy(ell), work(ell), internal(ell)]
        if (maxval(abs(totals - expected)) > tol) error stop "Node CSV rate order"
    end do
    read(unit, '(A)', iostat=status) line
    if (status >= 0) error stop "Unexpected extra node CSV row"
    close(unit, status="delete")

    ! 再次呼叫會重新配置輸出；同時覆蓋單層及零速率邊界情況。
    labels = 1
    heat = 0.0_dp
    work = 0.0_dp
    internal = 0.0_dp
    entropy = 0.0_dp
    call aggregate_layer_energetics( &
        labels, heat, work, internal, entropy, counts, lh, lw, lu, ls)
    if (size(counts) /= 1) error stop "Single-layer output size"
    if (counts(1) /= 5) error stop "Single-layer count"
    if (any(abs([lh(1), lw(1), lu(1), ls(1)]) > tol)) error stop "Zero rates"
    print *, "Layer aggregation and CSV test passed."
end program test_layer_aggregation
