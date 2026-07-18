from battery_designer.models import ElectricalLimits
from battery_designer.mos import select_mosfets


def test_mos_count_increases_with_current():
    low = select_mosfets(ElectricalLimits(continuous_current_a=1, peak_current_a=2, peak_duration_s=2, ambient_temp_c=25))
    high = select_mosfets(ElectricalLimits(continuous_current_a=5, peak_current_a=10, peak_duration_s=2, ambient_temp_c=25))
    assert high.package_count > low.package_count
    assert high.estimated_temp_rise_c <= 40
