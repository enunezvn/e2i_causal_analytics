"""
Automated Ontology Sync Script for CohortConstructor

This script handles quarterly updates of clinical codes from authoritative sources:
- ICD-10-CM (CMS)
- CPT/HCPCS (AMA) 
- NDC (FDA)
- LOINC (Regenstrief Institute)

Usage:
    python sync_clinical_codes.py --source icd10 --dry-run
    python sync_clinical_codes.py --source all --apply
"""

import yaml
import requests
from datetime import datetime
from typing import Dict, List, Set
from dataclasses import dataclass
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CodeUpdate:
    """Represents a proposed code change"""
    code: str
    description: str
    action: str  # 'add', 'deprecate', 'update'
    source: str
    effective_date: str
    notes: str = ""


class ClinicalCodeSynchronizer:
    """
    Syncs clinical codes from authoritative sources
    """
    
    def __init__(self, vocab_path: str = "cohort_vocabulary.yaml"):
        self.vocab_path = Path(vocab_path)
        self.vocab = self._load_vocabulary()
        self.proposals: List[CodeUpdate] = []
    
    def _load_vocabulary(self) -> dict:
        """Load current vocabulary"""
        with open(self.vocab_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _save_vocabulary(self, vocab: dict):
        """Save updated vocabulary"""
        with open(self.vocab_path, 'w') as f:
            yaml.dump(vocab, f, default_flow_style=False, sort_keys=False)
    
    def sync_icd10_codes(self) -> List[CodeUpdate]:
        """
        Sync ICD-10-CM codes from CMS
        
        CMS releases annual updates every October 1
        https://www.cms.gov/medicare/coding-billing/icd-10-codes
        """
        logger.info("Fetching ICD-10-CM updates from CMS...")
        
        # In production, fetch from CMS API or parse their XML releases
        # For demo, simulate new codes
        cms_codes = self._fetch_cms_icd10_updates()
        
        current_pattern = self.vocab['diagnosis_codes']['pattern']
        current_examples = set(self.vocab['diagnosis_codes']['examples'])
        
        updates = []
        for code_data in cms_codes:
            code = code_data['code']
            
            # Validate format
            if not self._matches_pattern(code, current_pattern):
                logger.warning(f"Skipping invalid ICD-10 code: {code}")
                continue
            
            # Check if new
            if code not in current_examples:
                updates.append(CodeUpdate(
                    code=code,
                    description=code_data['description'],
                    action='add',
                    source='CMS ICD-10-CM',
                    effective_date=code_data['effective_date'],
                    notes=f"New code for {code_data['condition']}"
                ))
        
        return updates
    
    def sync_ndc_codes(self) -> List[CodeUpdate]:
        """
        Sync NDC codes from FDA
        
        FDA updates daily via OpenFDA API
        https://open.fda.gov/apis/drug/ndc/
        """
        logger.info("Fetching NDC updates from FDA...")
        
        try:
            # Real implementation would use OpenFDA API
            response = requests.get(
                "https://api.fda.gov/drug/ndc.json",
                params={
                    'search': 'marketing_status:active',
                    'limit': 1000
                },
                timeout=30
            )
            response.raise_for_status()
            
            fda_data = response.json()
            updates = self._process_ndc_updates(fda_data)
            
            return updates
            
        except Exception as e:
            logger.error(f"Failed to fetch NDC codes: {e}")
            return []
    
    def sync_cpt_codes(self) -> List[CodeUpdate]:
        """
        Sync CPT codes from AMA
        
        AMA releases annual updates every January 1
        Note: CPT codes are copyrighted by AMA, requires license
        """
        logger.info("Fetching CPT updates from AMA...")
        
        # In production, would use licensed AMA CPT API
        # For now, return empty list
        logger.warning("CPT sync requires AMA license - skipping")
        return []
    
    def sync_loinc_codes(self) -> List[CodeUpdate]:
        """
        Sync LOINC codes from Regenstrief Institute
        
        Regenstrief releases biannually (June, December)
        https://loinc.org/downloads/
        """
        logger.info("Fetching LOINC updates from Regenstrief...")
        
        # In production, would download LOINC release files
        # For demo, simulate updates
        loinc_updates = self._fetch_loinc_updates()
        
        return loinc_updates
    
    def _fetch_cms_icd10_updates(self) -> List[dict]:
        """
        Simulate fetching ICD-10 updates from CMS
        
        In production, would:
        1. Download CMS release files (XML/Excel)
        2. Parse code definitions
        3. Compare with current vocabulary
        """
        # Simulated new codes for CSU/PNH/Breast Cancer
        return [
            {
                'code': 'L50.8',
                'description': 'Other urticaria',
                'condition': 'urticaria',
                'effective_date': '2025-10-01'
            },
            {
                'code': 'D59.51',
                'description': 'Paroxysmal nocturnal hemoglobinuria with crisis',
                'condition': 'PNH',
                'effective_date': '2025-10-01'
            },
            {
                'code': 'C50.911',
                'description': 'Malignant neoplasm of unspecified site of right female breast',
                'condition': 'breast cancer',
                'effective_date': '2025-10-01'
            }
        ]
    
    def _process_ndc_updates(self, fda_data: dict) -> List[CodeUpdate]:
        """Process NDC data from OpenFDA"""
        updates = []
        
        for result in fda_data.get('results', []):
            ndc = result.get('product_ndc')
            brand_name = result.get('brand_name')
            
            if ndc and brand_name:
                # Check if this is a relevant brand
                if brand_name in ['Remibrutinib', 'Fabhalta', 'Kisqali', 'Cosentyx']:
                    updates.append(CodeUpdate(
                        code=ndc,
                        description=f"{brand_name} - {result.get('dosage_form')}",
                        action='add',
                        source='FDA NDC',
                        effective_date=datetime.now().strftime('%Y-%m-%d'),
                        notes=f"Active marketing status"
                    ))
        
        return updates
    
    def _fetch_loinc_updates(self) -> List[CodeUpdate]:
        """Simulate fetching LOINC updates"""
        return [
            CodeUpdate(
                code='789-8',
                description='Erythrocytes [#/volume] in Blood by Automated count',
                action='add',
                source='LOINC',
                effective_date='2025-06-01',
                notes='Relevant for PNH monitoring'
            )
        ]
    
    def _matches_pattern(self, code: str, pattern: str) -> bool:
        """Validate code format against regex pattern"""
        import re
        return bool(re.match(pattern, code))
    
    def generate_proposal(self, updates: List[CodeUpdate]) -> dict:
        """
        Generate update proposal for human review
        """
        if not updates:
            return {'status': 'no_updates', 'updates': []}
        
        proposal = {
            'status': 'pending_review',
            'generated_at': datetime.now().isoformat(),
            'total_updates': len(updates),
            'updates_by_action': {
                'add': len([u for u in updates if u.action == 'add']),
                'deprecate': len([u for u in updates if u.action == 'deprecate']),
                'update': len([u for u in updates if u.action == 'update'])
            },
            'updates_by_source': {},
            'details': []
        }
        
        # Group by source
        for update in updates:
            source = update.source
            if source not in proposal['updates_by_source']:
                proposal['updates_by_source'][source] = 0
            proposal['updates_by_source'][source] += 1
            
            proposal['details'].append({
                'code': update.code,
                'description': update.description,
                'action': update.action,
                'source': update.source,
                'effective_date': update.effective_date,
                'notes': update.notes
            })
        
        return proposal
    
    def apply_updates(self, updates: List[CodeUpdate], version_bump: str = 'minor'):
        """
        Apply approved updates to vocabulary
        
        Args:
            updates: List of approved code updates
            version_bump: 'major', 'minor', or 'patch'
        """
        logger.info(f"Applying {len(updates)} updates to vocabulary...")
        
        # Group updates by target section
        icd10_adds = [u for u in updates if u.source.startswith('CMS') and u.action == 'add']
        ndc_adds = [u for u in updates if u.source.startswith('FDA') and u.action == 'add']
        loinc_adds = [u for u in updates if u.source.startswith('LOINC') and u.action == 'add']
        
        # Update diagnosis_codes examples
        if icd10_adds:
            current_examples = self.vocab['diagnosis_codes']['examples']
            for update in icd10_adds:
                if update.code not in current_examples:
                    current_examples.append(update.code)
                    logger.info(f"Added ICD-10 code: {update.code}")
        
        # Update drug_codes examples
        if ndc_adds:
            current_examples = self.vocab['drug_codes']['examples']
            for update in ndc_adds:
                if update.code not in current_examples:
                    current_examples.append(update.code)
                    logger.info(f"Added NDC code: {update.code}")
        
        # Update lab_codes examples
        if loinc_adds:
            current_examples = self.vocab['lab_codes']['examples']
            for update in loinc_adds:
                if update.code not in current_examples:
                    current_examples.append(update.code)
                    logger.info(f"Added LOINC code: {update.code}")
        
        # Bump version
        old_version = self._get_current_version()
        new_version = self._bump_version(old_version, version_bump)
        
        # Update vocabulary file header
        self._update_version_in_header(new_version)
        
        # Save updated vocabulary
        self._save_vocabulary(self.vocab)
        
        logger.info(f"✅ Vocabulary updated: {old_version} → {new_version}")
        logger.info(f"Applied {len(updates)} updates")
        
        # Generate changelog entry
        self._generate_changelog_entry(updates, old_version, new_version)
    
    def _get_current_version(self) -> str:
        """Extract version from YAML header comment"""
        with open(self.vocab_path, 'r') as f:
            first_line = f.readline()
            # Assumes format: "# COHORT CONSTRUCTOR DOMAIN VOCABULARY v1.0.0"
            import re
            match = re.search(r'v(\d+\.\d+\.\d+)', first_line)
            return match.group(1) if match else '1.0.0'
    
    def _bump_version(self, version: str, bump_type: str) -> str:
        """Bump semantic version"""
        major, minor, patch = map(int, version.split('.'))
        
        if bump_type == 'major':
            return f"{major + 1}.0.0"
        elif bump_type == 'minor':
            return f"{major}.{minor + 1}.0"
        else:  # patch
            return f"{major}.{minor}.{patch + 1}"
    
    def _update_version_in_header(self, new_version: str):
        """Update version number in YAML header"""
        with open(self.vocab_path, 'r') as f:
            content = f.read()
        
        import re
        updated_content = re.sub(
            r'v\d+\.\d+\.\d+',
            f'v{new_version}',
            content,
            count=1
        )
        
        with open(self.vocab_path, 'w') as f:
            f.write(updated_content)
    
    def _generate_changelog_entry(self, updates: List[CodeUpdate], old_version: str, new_version: str):
        """Generate changelog entry"""
        changelog_path = Path('ONTOLOGY_CHANGELOG.md')
        
        entry = f"\n## [{new_version}] - {datetime.now().strftime('%Y-%m-%d')}\n\n"
        entry += f"### Clinical Code Updates\n\n"
        
        # Group by source
        by_source = {}
        for update in updates:
            if update.source not in by_source:
                by_source[update.source] = []
            by_source[update.source].append(update)
        
        for source, source_updates in by_source.items():
            entry += f"#### {source}\n"
            for update in source_updates:
                entry += f"- **{update.action.upper()}**: `{update.code}` - {update.description}\n"
            entry += "\n"
        
        # Append to changelog
        if changelog_path.exists():
            with open(changelog_path, 'r') as f:
                existing = f.read()
            with open(changelog_path, 'w') as f:
                f.write(entry + existing)
        else:
            with open(changelog_path, 'w') as f:
                f.write(f"# Ontology Changelog\n{entry}")
        
        logger.info(f"📝 Changelog updated: {changelog_path}")


class OntologyValidator:
    """
    Validates ontology updates before merge
    """
    
    def __init__(self, vocab_path: str):
        self.vocab_path = Path(vocab_path)
    
    def validate_all(self) -> bool:
        """Run all validation checks"""
        checks = [
            ('YAML Syntax', self.validate_yaml_syntax),
            ('Version Bump', self.validate_version_bump),
            ('No Duplicates', self.validate_no_duplicates),
            ('Pattern Validity', self.validate_patterns),
            ('Alias Consistency', self.validate_aliases),
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            try:
                check_func()
                logger.info(f"✅ {check_name}: PASSED")
            except Exception as e:
                logger.error(f"❌ {check_name}: FAILED - {e}")
                all_passed = False
        
        return all_passed
    
    def validate_yaml_syntax(self):
        """Ensure valid YAML syntax"""
        with open(self.vocab_path, 'r') as f:
            yaml.safe_load(f)
    
    def validate_version_bump(self):
        """Ensure version was bumped"""
        # Compare with main branch version
        # For demo, just check version exists
        with open(self.vocab_path, 'r') as f:
            first_line = f.readline()
            if 'v' not in first_line:
                raise ValueError("Version number not found in header")
    
    def validate_no_duplicates(self):
        """Check for duplicate values in lists"""
        vocab = yaml.safe_load(open(self.vocab_path))
        
        for section, data in vocab.items():
            if isinstance(data, dict) and 'values' in data:
                values = data['values']
                if len(values) != len(set(values)):
                    duplicates = set([v for v in values if values.count(v) > 1])
                    raise ValueError(f"Duplicate values in {section}: {duplicates}")
    
    def validate_patterns(self):
        """Validate regex patterns and examples"""
        vocab = yaml.safe_load(open(self.vocab_path))
        
        import re
        for section, data in vocab.items():
            if isinstance(data, dict) and 'pattern' in data:
                pattern = data['pattern']
                examples = data.get('examples', [])
                
                for example in examples:
                    if not re.match(pattern, example):
                        raise ValueError(
                            f"Example '{example}' doesn't match pattern '{pattern}' in {section}"
                        )
    
    def validate_aliases(self):
        """Check alias consistency"""
        vocab = yaml.safe_load(open(self.vocab_path))
        
        aliases = vocab.get('aliases', {})
        for canonical, alias_list in aliases.items():
            if not isinstance(alias_list, list):
                raise ValueError(f"Aliases for '{canonical}' must be a list")
            
            # Check for self-reference
            if canonical.lower() in [a.lower() for a in alias_list]:
                raise ValueError(f"Canonical term '{canonical}' should not be in its own aliases")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sync clinical codes to ontology')
    parser.add_argument('--source', choices=['icd10', 'ndc', 'cpt', 'loinc', 'all'], required=True)
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying')
    parser.add_argument('--apply', action='store_true', help='Apply changes to vocabulary')
    parser.add_argument('--vocab-path', default='cohort_vocabulary.yaml')
    
    args = parser.parse_args()
    
    syncer = ClinicalCodeSynchronizer(vocab_path=args.vocab_path)
    
    # Collect updates based on source
    all_updates = []
    if args.source in ['icd10', 'all']:
        all_updates.extend(syncer.sync_icd10_codes())
    if args.source in ['ndc', 'all']:
        all_updates.extend(syncer.sync_ndc_codes())
    if args.source in ['cpt', 'all']:
        all_updates.extend(syncer.sync_cpt_codes())
    if args.source in ['loinc', 'all']:
        all_updates.extend(syncer.sync_loinc_codes())
    
    # Generate proposal
    proposal = syncer.generate_proposal(all_updates)
    
    if proposal['status'] == 'no_updates':
        logger.info("✅ No updates needed - vocabulary is current")
        return
    
    # Save proposal for review
    proposal_path = Path(f"proposals/{args.source}_update_{datetime.now().strftime('%Y%m%d')}.yaml")
    proposal_path.parent.mkdir(exist_ok=True)
    with open(proposal_path, 'w') as f:
        yaml.dump(proposal, f)
    
    logger.info(f"📋 Proposal saved: {proposal_path}")
    logger.info(f"Total updates: {proposal['total_updates']}")
    logger.info(f"  - Add: {proposal['updates_by_action']['add']}")
    logger.info(f"  - Deprecate: {proposal['updates_by_action']['deprecate']}")
    logger.info(f"  - Update: {proposal['updates_by_action']['update']}")
    
    if args.dry_run:
        logger.info("🔍 DRY RUN - No changes applied")
        logger.info(f"Review proposal: {proposal_path}")
        logger.info("To apply, run with --apply flag")
        return
    
    if args.apply:
        # Apply updates
        syncer.apply_updates(all_updates, version_bump='minor')
        
        # Validate
        validator = OntologyValidator(args.vocab_path)
        if validator.validate_all():
            logger.info("✅ All validation checks passed")
        else:
            logger.error("❌ Validation failed - please review errors")
            return
        
        logger.info("✅ Ontology sync complete!")
        logger.info("Next steps:")
        logger.info("  1. Review changes: git diff cohort_vocabulary.yaml")
        logger.info("  2. Run regression tests: pytest tests/test_query_regression.py")
        logger.info("  3. Create PR: git checkout -b chore/ontology-sync-{source}")


if __name__ == '__main__':
    main()
