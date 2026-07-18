from pathlib import Path

import pytest

from battery_designer.catalog import IcCatalog
from battery_designer.errors import DesignError
from battery_designer.generator import DesignGenerator
from battery_designer.kicad import KicadPipeline


def test_preview_is_generated_but_manufacturing_requires_template(common_spec, tmp_path: Path):
    device = IcCatalog(Path("data/ic_catalog"), tmp_path / "cache").resolve("DW01-G")
    generator = DesignGenerator(KicadPipeline())
    result = generator.generate_preview(common_spec, device, tmp_path)
    assert result["stage"] == "preview_ready"
    assert (tmp_path / "output/preview/mechanical_front.svg").exists()
    with pytest.raises(DesignError) as error:
        generator.generate_manufacturing(common_spec, device, tmp_path, approved=True)
    assert error.value.code == "IC_TEMPLATE_NOT_READY"
