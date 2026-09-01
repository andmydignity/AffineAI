import torch
from affine_ai.core.growth import StickBreakingGrowthController
from affine_ai.core.type_codebook import LatentTypeCodebook, rank_windows_by_info_gain
from affine_ai.core.rls_head import RLSPredictiveHead

def test_growth_controller():
    ctrl = StickBreakingGrowthController(alpha0=1.0, threshold=0.2, window=16)
    confident = torch.softmax(torch.randn(32, 8) * 3, dim=-1)
    for _ in range(20):
        ctrl.update(confident)
    should, mass = ctrl.should_grow()
    assert not should
    ctrl2 = StickBreakingGrowthController(alpha0=2.0, threshold=0.05, window=16)
    diffuse = torch.full((32, 8), 1/8)
    for _ in range(20):
        ctrl2.update(diffuse)
    should2, mass2 = ctrl2.should_grow()
    assert should2
    assert mass2 > 0.1

def test_type_codebook():
    torch.manual_seed(0)
    cb = LatentTypeCodebook(dim=8, max_types=8, birth_threshold=1.0, merge_threshold=0.5)
    # Initially 1 type, scale 0 -> enrichment is identity
    lat = torch.randn(2, 4, 8)
    enriched, ids = cb(lat)
    assert enriched.shape == lat.shape
    assert ids.shape == (2, 4)
    # Far latents spawn new types
    far = torch.randn(4, 8) * 5 + 10
    cb.update(far.unsqueeze(0).expand(2, -1, -1))
    assert int(cb.num_types.item()) > 1
    # BMR merge duplicate types
    cb.means[0] = torch.zeros(8)
    cb.means[1] = torch.zeros(8) + 0.01
    cb.counts[0] = 10
    cb.counts[1] = 10
    cb.num_types.fill_(2)
    merged = cb.bmr_merge()
    assert merged == 1
    assert int(cb.num_types.item()) == 1

def test_info_gain_ranking():
    head = RLSPredictiveHead(d_model=8, vocab_size=4)
    h_seen = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))
    head.update(h_seen.unsqueeze(0), y.unsqueeze(0))
    windows = [torch.randn(2, 4, 8) for _ in range(3)]
    ranked = rank_windows_by_info_gain(head, windows)
    assert set(ranked) == {0, 1, 2}
    assert len(ranked) == 3
    for idx in ranked:
        assert 0 <= idx < 3

def test_type_codebook_zero_init_preserving():
    cb = LatentTypeCodebook(dim=8, max_types=4)
    lat = torch.randn(2, 4, 8)
    enriched, _ = cb(lat)
    assert torch.allclose(enriched, lat, atol=1e-6)
