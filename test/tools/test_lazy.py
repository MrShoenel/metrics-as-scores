from metrics_as_scores.tools.lazy import SelfResetLazy
from pytest import raises



def test_SelfResetLazy():
    def destroy(v):
        assert v == 42
    srl = SelfResetLazy(
        fn_create_val=lambda: 42,
        fn_destroy_val=destroy,
        reset_after=0.25)
    
    assert abs(srl.reset_after - 0.25) <= 1e-12
    srl.reset_after = 0.5
    assert abs(srl.reset_after - 0.5) <= 1e-12
    
    assert not srl.has_value
    assert not srl.has_value_volatile

    assert srl.value == 42
    assert srl.value_volatile == 42
    assert srl.value_future.result() == 42

    srl.unset_value()
    assert not srl.has_value
    assert not srl.has_value_volatile


    # Also check failure:
    srl = SelfResetLazy(
        fn_create_val=lambda: exec('raise Exception("TEST")'),
        fn_destroy_val=None,
        reset_after=0.5)
    
    with raises(Exception, match='TEST'):
        srl.value_future.result()
