"""Stage 3 invariants: selection, controls, scope and energy reconstruction."""

import unittest
import numpy as np
import pandas as pd
from scipy.constants import R

import project2_common as common
import project4_energy_memory as energy
import project4_retention_learning as learning


class EnergyReplayTests(unittest.TestCase):
    def candidates(self):
        n=80
        rows=pd.DataFrame(np.tile([.2]*5,(n,1)),columns=common.X_COLUMNS)
        rows['recipe_id']=[f'r{i:03d}' for i in range(n)]
        rows['split']='train';rows['eligible']=True
        rows['instability_label']=[1]*25+[0,1]*15+[0]*25
        rows[energy.ENERGY]=-1-np.arange(n)*.01
        return rows

    def test_highest_lowest_and_control_class_counts(self):
        rows=self.candidates()
        memories,status,ranked=energy.choose_energy_memories(rows,11)
        self.assertEqual(set(memories['highest_g20'].recipe_id),set(rows.head(20).recipe_id))
        self.assertEqual(set(memories['lowest_g20'].recipe_id),set(rows.tail(20).recipe_id))
        for policy,control in energy.PAIRS.items():
            self.assertEqual(len(memories[policy]),20)
            self.assertEqual(memories[policy].recipe_id.nunique(),20)
            self.assertEqual(memories[policy].instability_label.sum(),memories[control].instability_label.sum())
        self.assertEqual(memories['highest_g20'].instability_label.sum(),20)
        self.assertEqual(memories['lowest_g20'].instability_label.sum(),0)

    def test_bridge_uses_first_negative_and_skips_later_positives(self):
        rows=self.candidates(); ranked=energy.ranked_eligible(rows)
        selected,info=energy.bridge_memory(ranked)
        self.assertEqual(info['first_negative_rank'],26)
        self.assertEqual(set(selected[selected.instability_label.eq(1)].recipe_id),set(rows.iloc[15:25].recipe_id))
        self.assertEqual(set(selected[selected.instability_label.eq(0)].recipe_id),set(rows.iloc[list(range(25,44,2))].recipe_id))
        self.assertEqual(info['skipped_intervening_positive_count'],9)
        self.assertEqual(selected.instability_label.sum(),10)

    def test_infeasible_bridge_has_no_fallback(self):
        rows=self.candidates();rows.loc[3,'instability_label']=0
        memories,status,_=energy.choose_energy_memories(rows,11)
        self.assertNotIn('energy_bridge20',memories)
        self.assertNotIn('bridge_class_matched20',memories)
        self.assertTrue(all(s['status']=='unavailable' for s in status))
        self.assertEqual(status[0]['first_negative_rank'],4)
        self.assertEqual(len(memories),4)

    def test_no_negative_bridge_and_too_few_negatives(self):
        rows=self.candidates();rows['instability_label']=1
        self.assertIsNone(energy.bridge_memory(energy.ranked_eligible(rows))[0])
        rows.loc[25:29,'instability_label']=0
        self.assertIsNone(energy.bridge_memory(energy.ranked_eligible(rows))[0])

    def test_order_ties_and_ineligible_examples(self):
        rows=self.candidates();rows.loc[:2,energy.ENERGY]=0.;rows.loc[0,'eligible']=False
        a,_,_=energy.choose_energy_memories(rows,22)
        b,_,_=energy.choose_energy_memories(rows.sample(frac=1,random_state=901),22)
        for key in a:
            self.assertEqual(a[key].recipe_id.tolist(),b[key].recipe_id.tolist())
            self.assertNotIn('r000',a[key].recipe_id.tolist())
        self.assertEqual(energy.ranked_eligible(rows).iloc[0].recipe_id,'r001')

    def test_heldout_candidates_rejected(self):
        rows=self.candidates();rows.loc[0,'split']='test'
        with self.assertRaisesRegex(ValueError,'training'):
            energy.choose_energy_memories(rows,11)

    def test_total_energy_is_excess_plus_ideal(self):
        class FixedExcess:
            def predict(self,x):return np.full(len(x),1.25)
        rows=self.candidates().iloc[:3]
        result,_=energy.predicted_energies(FixedExcess(),rows)
        np.testing.assert_allclose(result[energy.ENERGY],1.25-R*298.15*np.log(5)/1000)
        outside=rows.copy();outside.loc[0,common.X_COLUMNS]=[.6,.1,.1,.1,.1]
        with self.assertRaisesRegex(ValueError,'in-domain'):
            energy.predicted_energies(FixedExcess(),outside)

    def test_energy_does_not_become_a_classifier_feature(self):
        rows=self.candidates();changed=rows.copy();changed[energy.ENERGY]=np.arange(len(rows))*1000
        for columns in [common.COMPOSITION,common.feature_groups()['b_molecular']]:
            np.testing.assert_array_equal(learning.features(rows,columns),learning.features(changed,columns))


if __name__=='__main__':unittest.main()
