@testset "sparse stuff" begin
    v1 = sprandn(100, 0.5)
    v2 = sprandn(100, 0.5)

    val1 = PBVI.sparse_vec_norm_diff(v1, v2)
    val2 = sum(abs2, v1 .- v2)
    @test val1 ≈ val2

end
