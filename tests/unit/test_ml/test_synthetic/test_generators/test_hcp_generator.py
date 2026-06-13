"""
Tests for HCPGenerator.

Tests HCP profile generation.
"""

import pandas as pd
import pytest

from src.ml.synthetic.config import Brand, PracticeTypeEnum, RegionEnum, SpecialtyEnum
from src.ml.synthetic.generators import GeneratorConfig, HCPGenerator


class TestHCPGenerator:
    """Test suite for HCPGenerator."""

    @pytest.fixture
    def generator(self):
        """Create an HCPGenerator instance."""
        config = GeneratorConfig(
            seed=42,
            n_records=100,
        )
        return HCPGenerator(config)

    @pytest.fixture
    def brand_specific_generator(self):
        """Create a brand-specific generator."""
        config = GeneratorConfig(
            seed=42,
            n_records=100,
            brand=Brand.REMIBRUTINIB,
        )
        return HCPGenerator(config)

    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator.config.n_records == 100
        assert generator.config.seed == 42
        assert generator.entity_type == "hcp_profiles"

    def test_generate_correct_count(self, generator):
        """Test that generator produces correct number of records."""
        df = generator.generate()
        assert len(df) == 100

    def test_generate_correct_columns(self, generator):
        """Test that generated DataFrame has correct columns."""
        df = generator.generate()

        required_columns = [
            "hcp_id",
            "npi",
            "specialty",
            "practice_type",
            "geographic_region",
            "years_experience",
            "academic_hcp",
            "total_patient_volume",
            "brand",
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_hcp_ids_unique(self, generator):
        """Test that HCP IDs are unique."""
        df = generator.generate()
        assert df["hcp_id"].nunique() == len(df)

    def test_npi_format(self, generator):
        """Test that NPIs are 10-digit strings."""
        df = generator.generate()

        for npi in df["npi"]:
            assert len(npi) == 10
            assert npi.isdigit()

    def test_specialty_values_valid(self, generator):
        """Test that specialty values are valid enums."""
        df = generator.generate()
        valid_specialties = {s.value for s in SpecialtyEnum}

        for specialty in df["specialty"].unique():
            assert specialty in valid_specialties, f"Invalid specialty: {specialty}"

    def test_practice_type_values_valid(self, generator):
        """Test that practice type values are valid enums."""
        df = generator.generate()
        valid_types = {p.value for p in PracticeTypeEnum}

        for practice_type in df["practice_type"].unique():
            assert practice_type in valid_types, f"Invalid practice type: {practice_type}"

    def test_region_values_valid(self, generator):
        """Test that region values are valid enums."""
        df = generator.generate()
        valid_regions = {r.value for r in RegionEnum}

        for region in df["geographic_region"].unique():
            assert region in valid_regions, f"Invalid region: {region}"

    def test_years_experience_range(self, generator):
        """Test that years of experience is within valid range."""
        df = generator.generate()

        assert df["years_experience"].min() >= 2
        assert df["years_experience"].max() < 40

    def test_academic_hcp_binary(self, generator):
        """Test that academic_hcp is binary."""
        df = generator.generate()

        assert set(df["academic_hcp"].unique()).issubset({0, 1})

    def test_patient_volume_range(self, generator):
        """Test that patient volume is within valid range."""
        df = generator.generate()

        assert df["total_patient_volume"].min() >= 50
        assert df["total_patient_volume"].max() <= 600

    def test_brand_specific_generation(self, brand_specific_generator):
        """Test that brand-specific generator produces correct brand."""
        df = brand_specific_generator.generate()

        assert all(df["brand"] == Brand.REMIBRUTINIB.value)

    def test_brand_specialty_alignment(self, brand_specific_generator):
        """Test that specialties align with brand."""
        df = brand_specific_generator.generate()

        # Remibrutinib targets dermatology, allergy_immunology, rheumatology (CSU indication)
        expected_specialties = {
            SpecialtyEnum.DERMATOLOGY.value,
            SpecialtyEnum.ALLERGY_IMMUNOLOGY.value,
            SpecialtyEnum.RHEUMATOLOGY.value,
        }

        for specialty in df["specialty"].unique():
            assert specialty in expected_specialties

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        config = GeneratorConfig(seed=42, n_records=50)

        gen1 = HCPGenerator(config)
        df1 = gen1.generate()

        gen2 = HCPGenerator(config)
        df2 = gen2.generate()

        pd.testing.assert_frame_equal(df1, df2)

    def test_generate_batched(self):
        """Test batch generation."""
        config = GeneratorConfig(
            seed=42,
            n_records=250,
            batch_size=100,
        )
        generator = HCPGenerator(config)

        batches = list(generator.generate_batched())

        assert len(batches) == 3  # 100 + 100 + 50
        assert len(batches[0]) == 100
        assert len(batches[1]) == 100
        assert len(batches[2]) == 50

    def test_generate_with_result(self, generator):
        """Test generation with result metadata."""
        result = generator.generate_with_result()

        assert result.entity_type == "hcp_profiles"
        assert result.n_records == 100
        assert result.generation_time > 0
        assert result.is_valid


class TestHCPGeneratorDistributions:
    """Test statistical distributions in generated data."""

    @pytest.fixture
    def large_generator(self):
        """Create generator with large sample for distribution tests."""
        config = GeneratorConfig(
            seed=42,
            n_records=5000,
        )
        return HCPGenerator(config)

    def test_practice_type_distribution(self, large_generator):
        """Test that practice type follows expected distribution."""
        df = large_generator.generate()
        dist = df["practice_type"].value_counts(normalize=True)

        # Expected: academic=0.25, community=0.50, private=0.25
        # Allow 5% deviation for randomness
        assert abs(dist.get(PracticeTypeEnum.COMMUNITY.value, 0) - 0.50) < 0.10

    def test_region_distribution(self, large_generator):
        """Test that regions follow expected distribution."""
        df = large_generator.generate()
        dist = df["geographic_region"].value_counts(normalize=True)

        # South should be largest (~38%)
        assert dist.get(RegionEnum.SOUTH.value, 0) > 0.30

    def test_academic_hcp_rate_by_practice(self, large_generator):
        """Test that academic flag correlates with practice type."""
        df = large_generator.generate()

        academic_practice = df[df["practice_type"] == PracticeTypeEnum.ACADEMIC.value]
        other_practice = df[df["practice_type"] != PracticeTypeEnum.ACADEMIC.value]

        academic_rate_in_academic = academic_practice["academic_hcp"].mean()
        academic_rate_in_other = other_practice["academic_hcp"].mean()

        # Academic practice should have higher academic HCP rate
        assert academic_rate_in_academic > academic_rate_in_other


class TestHCPTerritoryAssignment:
    """Territory/sales-rep assignment on hcp_profiles.

    The territory_metrics panel/ETL keys off hcp_profiles.territory_id (the ETL's
    `territories` CTE = SELECT DISTINCT territory_id WHERE territory_id IS NOT NULL).
    Historically the generator left territory_id/sales_rep_id NULL, so the panel
    was always empty. Territories nest within geographic_region: each of the 4
    regions is split into TERRITORIES_PER_REGION territories.
    """

    def _gen(self, seed=42, n=400, brand=None):
        cfg = GeneratorConfig(seed=seed, n_records=n, brand=brand)
        return HCPGenerator(cfg).generate()

    def test_every_hcp_has_territory_and_rep(self):
        df = self._gen()
        assert df["territory_id"].notna().all()
        assert df["sales_rep_id"].notna().all()
        assert (df["territory_id"].astype(str).str.len() > 0).all()

    def test_territory_nests_within_region(self):
        # territory_id is "<region>-T<NN>"; its region prefix must equal the row's
        # geographic_region (no cross-region leakage).
        df = self._gen()
        prefix = df["territory_id"].astype(str).str.rsplit("-T", n=1).str[0]
        assert (prefix == df["geographic_region"].astype(str)).all()

    def test_territory_id_format_and_count(self):
        df = self._gen(n=2000)
        assert df["territory_id"].astype(str).str.match(r"^[a-z]+-T\d{2}$").all()
        # 4 regions x K territories; never more distinct territories than that.
        k = HCPGenerator.TERRITORIES_PER_REGION
        assert df["territory_id"].nunique() <= 4 * k
        # Per region, no more than K distinct territories.
        per_region = df.groupby("geographic_region")["territory_id"].nunique()
        assert (per_region <= k).all()

    def test_sales_rep_derived_from_territory(self):
        df = self._gen()
        assert (df["sales_rep_id"].astype(str) == "REP-" + df["territory_id"].astype(str)).all()

    def test_assignment_is_deterministic(self):
        a = self._gen(seed=7)
        b = self._gen(seed=7)
        assert a["territory_id"].tolist() == b["territory_id"].tolist()
