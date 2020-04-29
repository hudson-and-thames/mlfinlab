"""
Tests the unit functions in ef3m.py for calculating fitting a mixture of 2 Gaussian distributions.
"""

import unittest
import numpy as np
import pandas as pd
from scipy.special import comb

from mlfinlab.bet_sizing.ef3m import (M2N, centered_moment, raw_moment, most_likely_parameters, iter_4_jit, iter_5_jit)


class TestM2NConstructor(unittest.TestCase):
    """
    Tests the constructor method of the M2N class.
    """

    def test_m2n_constructor(self):
        """
        Tests that the constructor of the M2N class executes properly.
        """
        moments_test = [1, 2, 3, 4, 5]
        m2n_test = M2N(moments_test)
        # Confirm that the initial attributes get set properly.
        self.assertEqual(m2n_test.moments, moments_test)
        self.assertEqual(m2n_test.new_moments, [0, 0, 0, 0, 0])
        self.assertEqual(m2n_test.parameters, [0, 0, 0, 0, 0])
        self.assertEqual(m2n_test.error, sum([moments_test[i]**2 for i in range(len(moments_test))]))


class TestM2NGetMoments(unittest.TestCase):
    """
    Tests the 'get_moments' method of the M2N class.
    """
    def test_get_moments(self):
        """
        Tests the 'get_moments' method of the M2N class.
        """
        u_1, u_2, s_1, s_2, p_1 = [2.1, 4.3, 1.1, 0.7, 0.3]
        p_2 = 1 - p_1
        m_1 = p_1*u_1 + p_2*u_2
        m_2 = p_1*(s_1**2 + u_1**2) + p_2*(s_2**2 + u_2**2)
        m_3 = p_1*(3*s_1**2*u_1 + u_1**3) + p_2*(3*s_2**2*u_2 + u_2**3)
        m_4 = p_1*(3*s_1**4 + 6*s_1**2*u_1**2 + u_1**4) + p_2*(3*s_2**4 + 6*s_2**2*u_2**2 + u_2**4)
        m_5 = p_1*(15*s_1**4*u_1 + 10*s_1**2*u_1**3 + u_1**5) + p_2*(15*s_2**4*u_2 + 10*s_2**2*u_2**3 + u_2**5)
        test_params = [u_1, u_2, s_1, s_2, p_1]
        test_mmnts = [m_1, m_2, m_3, m_4, m_5]
        # Create M2N object.
        m2n_test = M2N(test_mmnts)
        # Check self-return method.
        m2n_test.get_moments(test_params, return_result=False)
        self.assertEqual(test_mmnts, m2n_test.new_moments)
        # Check the function when 'return_value' is True.
        result_mmnts = m2n_test.get_moments(test_params, return_result=True)
        self.assertEqual(test_mmnts, result_mmnts)


class TestM2NIter4(unittest.TestCase):
    """
    Tests the 'iter_4' method of the M2N class.
    """
    def test_iter_4_validity_check_1(self):
        """
        Tests 'iter_4' method's 'Validity check 1' breakpoint condition.
        """
        moments_test = [1, 2, 3, 4, 5]
        m2n_test = M2N(moments_test)
        param_results = m2n_test.iter_4(3, 1)
        self.assertTrue(not param_results)

    def test_iter_4_validity_check_2(self):
        """
        Tests 'iter_4' method's 'Validity check 2' breakpoint condition.
        """
        moments_test = [2, 2, 3, 4, 5]
        m2n_test = M2N(moments_test)
        param_results = m2n_test.iter_4(1, 0.8)
        self.assertTrue(not param_results)

    def test_iter_4_validity_check_3(self):
        """
        Tests 'iter_4' method's 'Validity check 3' breakpoint condition.
        """
        moments_test = [1.5, 2, 3, 4, 5]
        m2n_test = M2N(moments_test)
        param_results = m2n_test.iter_4(2, 0.7)
        self.assertTrue(not param_results)

    # Validity check 4 is not present since the condition appears to be unreachable.

    def test_iter_4_validity_check_5(self):
        """
        Tests 'iter_4' method's 'Validity check 5' breakpoint condition.
        """
        moments_test = [0.0, 0.1, 0.0, 0.0, 5]
        m2n_test = M2N(moments_test)
        param_results = m2n_test.iter_4(0.1, 0.5)
        self.assertTrue(not param_results)

    def test_iter_4_validity_check_6(self):
        """
        Tests 'iter_4' method's 'Validity check 6' breakpoint condition.
        """
        moments_test = [0.0, 0.1, 0.0, 0.0, 5]
        m2n_test = M2N(moments_test)
        param_results = m2n_test.iter_4(0.1, 0.25)
        self.assertTrue(not param_results)

    def test_iter_4_success(self):
        """
        Tests 'iter_4' method for successful execution.
        """
        moments_test = [0.7, 2.6, 0.4, 25, -59.8]
        m2n_test = M2N(moments_test)
        param_results = m2n_test.iter_4(1, 0.2)
        self.assertTrue(len(param_results) == 5)


class TestM2NIter5(unittest.TestCase):
    """
    Tests the 'iter_5' method of the M2N class.
    """
    def test_iter_5_validity_check_1(self):
        """
        Tests 'iter_5' method's 'Validity check 1' breakpoint condition.
        """
        moments_test = [0.0, 0.0, 0.0, 0.0, 0.0]
        mu_2_test, p_1_test = 0.0, 0.05
        m2n_test = M2N(moments_test)
        param_results = m2n_test.iter_5(mu_2_test, p_1_test)
        self.assertTrue(not param_results)

    def test_iter_5_validity_check_2(self):
        """
        Tests 'iter_5' method's 'Validity check 2' breakpoint condition.
        """
        moments_test = [0.0, 0.0, 0.0, 0.0, 0.0]
        mu_2_test, p_1_test = 0.1, 0.05
        m2n_test = M2N(moments_test)
        param_results = m2n_test.iter_5(mu_2_test, p_1_test)
        self.assertTrue(not param_results)

    def test_iter_5_validity_check_3(self):
        """
        Tests 'iter_5' method's 'Validity check 3' breakpoint condition.
        """
        moments_test = [0.0, 0.0, 0.1, 0.0, 0.0]
        mu_2_test, p_1_test = 0.1, 0.2
        m2n_test = M2N(moments_test)
        param_results = m2n_test.iter_5(mu_2_test, p_1_test)
        self.assertTrue(not param_results)

    # Validity check 4 is not present since the condition appears to be unreachable.

    def test_iter_5_validity_check_5(self):
        """
        Tests 'iter_5' method's 'Validity check 5' breakpoint condition.
        """
        moments_test = [0.0, 0.1, 0.0, 0.0, 0.0]
        mu_2_test, p_1_test = 0.1, 0.99999
        m2n_test = M2N(moments_test)
        param_results = m2n_test.iter_5(mu_2_test, p_1_test)
        self.assertTrue(not param_results)

    def test_iter_5_validity_check_6(self):
        """
        Tests 'iter_5' method's 'Validity check 6' breakpoint condition.
        """
        moments_test = [0.0, 0.1, 0.0, 0.0, 0.0]
        mu_2_test, p_1_test = 0.1, 0.95
        m2n_test = M2N(moments_test)
        param_results = m2n_test.iter_5(mu_2_test, p_1_test)
        self.assertTrue(not param_results)

    def test_iter_5_validity_check_7(self):
        """
        Tests 'iter_5' method's 'Validity check 7' breakpoint condition.
        """
        moments_test = [0.0, 0.1, 0.1, 0.0, 0.2]
        mu_2_test, p_1_test = 0.4, 0.95
        m2n_test = M2N(moments_test)
        param_results = m2n_test.iter_5(mu_2_test, p_1_test)
        self.assertTrue(not param_results)

    def test_iter_5_validity_check_8(self):
        """
        Tests 'iter_5' method's 'Validity check 8' breakpoint condition.
        """
        moments_test = [1.7486117351052706, 12.30094642908807, 44.14804719610457, 301.66990880582324, 1389.7073066865096]
        mu_2_test, p_1_test = 8.927498436080297, -1910484717784700.2
        m2n_test = M2N(moments_test)
        param_results = m2n_test.iter_5(mu_2_test, p_1_test)
        self.assertTrue(not param_results)

    def test_iter_5_validity_check_9(self):
        """
        Tests 'iter_5' method's 'Validity check 9' breakpoint condition.
        """
        moments_test = [1.7465392043495434, 12.32010406019726, 44.3090981635415, 302.3152423573811, 1403.0640473698527]
        mu_2_test, p_1_test = 1.8733475857864539, 0.019291066689915537
        m2n_test = M2N(moments_test)
        param_results = m2n_test.iter_5(mu_2_test, p_1_test)
        self.assertTrue(not param_results)

    def test_iter_5_success(self):
        """
        Tests 'iter_5' method for successful execution.
        """
        moments_test = [0.7, 2.6, 0.4, 25, -59.8]
        mu_2_test, p_1_test = 0.8642146104188053, 0.03296760034110158
        m2n_test = M2N(moments_test)
        param_results = m2n_test.iter_5(mu_2_test, p_1_test)
        self.assertTrue(len(param_results) == 5)


class TestM2NFit(unittest.TestCase):
    """
    Tets the 'fit' method of the M2N class.
    """
    def test_fit_variant_1(self):
        """
        Tests the 'fit' method of the M2N class, using variant 1.
        """
        moments_test = [0.7, 2.6, 0.4, 25, -59.8]
        mu_2_test = 1
        epsilon_test = 1e-5
        factor_test = 5
        n_runs_test = 5
        variant_test = 1
        max_iter_test = 10_000
        m2n_test = M2N(moments_test, epsilon_test, factor_test, n_runs_test, variant_test, max_iter_test)
        m2n_test.fit(mu_2_test)
        self.assertTrue(len(m2n_test.parameters) == 5)

    def test_fit_variant_2(self):
        """
        Tests the 'fit' method of the M2N class, using variant 2.
        """
        moments_test = [0.7, 2.6, 0.4, 25, -59.8]
        mu_2_test = 1
        epsilon_test = 1e-5
        factor_test = 5
        n_runs_test = 5
        variant_test = 2
        max_iter_test = 10_000
        m2n_test = M2N(moments_test, epsilon_test, factor_test, n_runs_test, variant_test, max_iter_test)
        m2n_test.fit(mu_2_test)
        self.assertTrue(len(m2n_test.parameters) == 5)

    def test_fit_variant_value_error(self):
        """
        Tests that the 'fit' method throws a ValueError if an invalid value is passed to argument 'variant'.
        """
        moments_test = [0.7, 2.6, 0.4, 25, -59.8]
        mu_2_test = 1
        epsilon_test = 1e-5
        factor_test = 5
        n_runs_test = 5
        variant_test = 3
        max_iter_test = 10_000
        m2n_test = M2N(moments_test, epsilon_test, factor_test, n_runs_test, variant_test, max_iter_test)
        self.assertRaises(ValueError, m2n_test.fit, mu_2_test)

    def test_fit_success_via_error(self):
        """
        Tests that the 'fit' method successfully exits due to a low error being reached.
        """
        moments_test = [0.7, 2.6, 0.4, 25, -59.8]
        mu_2_test = 1
        epsilon_test = 1e-5
        factor_test = 5
        n_runs_test = 5
        variant_test = 1
        max_iter_test = 10_000
        mu_2_test, epsilon_test, variant_test, max_iter_test = 1, 1e-5, 1, 10_000
        m2n_test = M2N(moments_test, epsilon_test, factor_test, n_runs_test, variant_test, max_iter_test)
        m2n_test.error = 1e6
        m2n_test.fit(mu_2_test)
        self.assertTrue(len(m2n_test.parameters) == 5)

    def test_fit_success_via_epsilon(self):
        """
        Tests that the 'fit' method successfully exits due to p_1 converging.
        """
        moments_test = [0.7, 2.6, 0.4, 25, -59.8]
        mu_2_test = 1
        epsilon_test = 1e12
        factor_test = 5
        n_runs_test = 5
        variant_test = 1
        max_iter_test = 10_000
        mu_2_test, epsilon_test, variant_test, max_iter_test = 1, 1e12, 1, 10_000
        np.random.seed(12)
        m2n_test = M2N(moments_test, epsilon_test, factor_test, n_runs_test, variant_test, max_iter_test)
        m2n_test.fit(mu_2_test)
        self.assertTrue(len(m2n_test.parameters) == 5)

    def test_fit_success_via_max_iter(self):
        """
        Tests that the 'fit' method successfully exits due to the maximum number of iterations being reached.
        """
        moments_test = [0.7, 2.6, 0.4, 25, -59.8]
        np.random.seed(12)
        mu_2_test = 1
        epsilon_test = 1e-12
        factor_test = 5
        n_runs_test = 5
        variant_test = 1
        max_iter_test = 1
        mu_2_test, epsilon_test, variant_test, max_iter_test = 1, 1e-12, 1, 1
        m2n_test = M2N(moments_test, epsilon_test, factor_test, n_runs_test, variant_test, max_iter_test)
        m2n_test.fit(mu_2=mu_2_test)
        self.assertTrue(len(m2n_test.parameters) == 5)


class TestM2NEF3M(unittest.TestCase):
    """
    Tests the EF3M algorithms of the M2N module.
    """

    def test_ef3m_variant_1(self):
        """
        Tests the 'iter_4_jit' function of the M2N module (using variant 1).
        """
        mu_2 = -0.5876905479546004
        p_1 = 0.020950069267730465
        moments_test = [
            -0.59,
            9.830000000000002,
            -19.922000000000004,
            264.254,
        ]
        expected_parameters = [
            -0.6979265579594388,
            -0.5876905479546004,
            20.80638279572574,
            0.6488995559493259,
            0.0004662589937450317,
        ]
        param_list = iter_4_jit(mu_2, p_1, *moments_test)
        param_list = param_list.tolist()
        self.assertTrue(expected_parameters == param_list)

    def test_ef3m_variant_2(self):
        """
        Tests the 'iter_5_jit' function of the M2N module (using variant 2).
        """
        mu_2 = -0.07037328978510915
        p_1 = 0.09166890087206325
        moments_test = [
            -0.59,
            9.830000000000002,
            -19.922000000000004,
            264.254,
            -818.2100000000003,
        ]
        expected_parameters = [
            -5.738890150700704,
            0.1755905244444409,
            0.862038546663441,
            2.723656932895561,
            0.12317856407155195,
        ]
        param_list = iter_5_jit(mu_2, p_1, *moments_test)
        param_list = param_list.tolist()
        self.assertTrue(expected_parameters == param_list)


class TestM2NSingleFitLoop(unittest.TestCase):
    """
    Tests the 'single_fit_loop' method.
    """
    def test_single_fit_loop_return_type(self):
        """
        Tests that the 'single_fit_loop' method executes successfully.
        """
        moments_test = [0.7, 2.6, 0.4, 25, -59.8]
        epsilon_test = 1e-5
        factor_test = 5
        n_runs_test = 10
        variant_test = 2
        max_iter_test = 10_000
        epsilon_test, factor_test, variant_test, max_iter_test = 1e-5, 5, 2, 10_000
        np.random.seed(12)
        m2n_test = M2N(moments_test, epsilon_test, factor_test, n_runs_test, variant_test, max_iter_test)
        df_results = m2n_test.single_fit_loop()
        self.assertTrue(isinstance(df_results, pd.DataFrame))


class TestM2NMpFit(unittest.TestCase):
    """
    Tests the 'mp_fit' method.
    """
    def test_mp_fit_return_type(self):
        """
        Tests that the 'mp_fit' method executes successfully.
        """
        moments_test = [0.7, 2.6, 0.4, 25, -59.8]
        epsilon_test = 1e-5
        factor_test = 5
        n_runs_test = 10
        variant_test = 2
        max_iter_test = 10_000
        num_workers_test = 1
        epsilon_test, factor_test, n_runs_test, variant_test, max_iter_test, num_workers_test = 1e-5, 5, 10, 2, 10_000, 1
        m2n_test = M2N(moments_test, epsilon_test, factor_test, n_runs_test, variant_test, max_iter_test, num_workers_test)
        df_results = m2n_test.mp_fit()
        self.assertTrue(isinstance(df_results, pd.DataFrame))


class TestCenteredMoment(unittest.TestCase):
    """
    Tests the helper function 'centered_moment'.
    """
    def test_centered_moment_result(self):
        """
        Tests for the successful execution of the 'centered_moment' helper function.
        """
        raw_test = [0.701756, 2.591815, 0.450519, 24.689030, -57.756735]
        centered_5th_correct = 0
        for j in range(6):
            if j == 5:
                add_on = 1
            else:
                add_on = raw_test[5-j-1]
            centered_5th_correct += (-1)**j * int(comb(5, j)) * add_on * raw_test[0]**j
        centered_5th_test = centered_moment(raw_test, 5)
        self.assertAlmostEqual(centered_5th_test, centered_5th_correct, 7)


class TestRawMoment(unittest.TestCase):
    """
    Tests the helper function 'raw_moment'.
    """
    def test_raw_moment_result(self):
        """
        Tests for the successful execution of the 'raw_moment' helper function.
        """
        centered_test = [0.0, 2.11, -4.373999999999999, 30.803699999999996, -153.58572]
        raw_result = raw_moment(centered_test, 0.7)
        raw_correct = [0.7, 2.6, 0.4, 25, -59.8]
        self.assertTrue(np.allclose(raw_result, raw_correct, 1e-7))


class TestMostLikelyParameters(unittest.TestCase):
    """
    Tests the helper function 'most_likely_parameters'.
    """
    def test_most_likely_parameters_result(self):
        """
        Tests for the successful execution of the 'most_likely_parameters' function.
        """
        mu_1_list = [-2.074149682208028, -2.1464760973734522, -1.7318027625411423, -1.7799163398785354, -1.9766582333677596]
        mu_2_list = [0.9958122958772418, 0.9927128514876395, 1.013574632526087, 1.0065707257309104, 1.009533655971151]
        sigma_1_list = [1.9764097851543956, 1.9516780127056625, 2.080573657129795, 2.071328499049906, 1.9988591140726848]
        sigma_2_list = [1.002964090440232, 1.0054392587806025, 0.9872577865302316, 0.9909001363163131, 0.9971327048101786]
        p_1_list = [0.09668610445835334, 0.09379917992315062, 0.11351960785118335, 0.10993400151299484, 0.10264463363929438]
        df_test = pd.DataFrame.from_dict({'mu_1': mu_1_list,
                                          'mu_2': mu_2_list,
                                          'sigma_1': sigma_1_list,
                                          'sigma_2': sigma_2_list,
                                          'p_1': p_1_list})
        most_likely_correct = {'mu_1': -2.03765,
                               'mu_2': 1.00863,
                               'sigma_1': 1.9832,
                               'sigma_2': 1.00012,
                               'p_1': 0.09942}
        d_results = most_likely_parameters(data=df_test)
        self.assertTrue(np.allclose(list(d_results.values()), list(most_likely_correct.values()), 1e-7))

    def test_most_likely_parameters_list_arg(self):
        """
        Tests the helper function 'most_likely_parameters' when passing a list to 'ignore_columns'.
        """
        mu_1_list = [-2.074149682208028, -2.1464760973734522, -1.7318027625411423, -1.7799163398785354, -1.9766582333677596]
        mu_2_list = [0.9958122958772418, 0.9927128514876395, 1.013574632526087, 1.0065707257309104, 1.009533655971151]
        sigma_1_list = [1.9764097851543956, 1.9516780127056625, 2.080573657129795, 2.071328499049906, 1.9988591140726848]
        sigma_2_list = [1.002964090440232, 1.0054392587806025, 0.9872577865302316, 0.9909001363163131, 0.9971327048101786]
        p_1_list = [0.09668610445835334, 0.09379917992315062, 0.11351960785118335, 0.10993400151299484, 0.10264463363929438]
        df_test = pd.DataFrame.from_dict({'mu_1': mu_1_list,
                                          'mu_2': mu_2_list,
                                          'sigma_1': sigma_1_list,
                                          'sigma_2': sigma_2_list,
                                          'p_1': p_1_list})
        most_likely_correct = {'mu_1': -2.03765,
                               'mu_2': 1.00863,
                               'sigma_1': 1.9832,
                               'sigma_2': 1.00012,
                               'p_1': 0.09942}
        d_results = most_likely_parameters(data=df_test, ignore_columns=['error'])
        self.assertTrue(np.allclose(list(d_results.values()), list(most_likely_correct.values()), 1e-7))
