def dyck_paths(k):
    paths = []
    
    def generate(c1, c2, path):
        if c1 == k and c2 == k:
            paths.append(path)
            return

        if c1 < k:
            generate(c1 + 1, c2, path + [(c1 + 1, c2)])

        if c2 < c1:
            generate(c1, c2 + 1, path + [(c1, c2 + 1)])

    generate(0, 0, [(0, 0)])

    return paths


def G(l, k, sigma, w, N):
    result = sigma[l-k-1] * w[l]

    for j in range(1, k+1):
        result *= N[l-j] * N[l-j-1] * w[l-j]**2

    return result


def a(l, k, lamb, sigma, w, N):
    dyck_sum = 0.0

    for path in dyck_paths(k):
        path_product = 1.0

        for c1, c2 in path:
            path_product /= (
                lamb[l-c1] + lamb[l-1-c2]
            )

        dyck_sum += path_product

    return G(l, k, sigma, w, N) * dyck_sum