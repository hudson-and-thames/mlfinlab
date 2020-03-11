"""
Package based on the text book: Advances in Financial Machine Learning, by Marcos Lopez de Prado
"""
import webbrowser
import textwrap

import mlfinlab.cross_validation as cross_validation
import mlfinlab.data_structures as data_structures
import mlfinlab.multi_product as multi_product
import mlfinlab.filters.filters as filters
import mlfinlab.labeling.labeling as labeling
import mlfinlab.features.fracdiff as fracdiff
import mlfinlab.sample_weights as sample_weights
import mlfinlab.sampling as sampling
import mlfinlab.bet_sizing as bet_sizing
import mlfinlab.util as util
import mlfinlab.structural_breaks as structural_breaks
import mlfinlab.feature_importance as feature_importance
import mlfinlab.ensemble as ensemble
import mlfinlab.portfolio_optimization as portfolio_optimization
import mlfinlab.clustering as clustering


# Sponsorship notification
try:
    webbrowser.get('google-chrome').open_new('https://www.patreon.com/HudsonThames')
except webbrowser.Error as error:
    try:
        webbrowser.get('firefox').open_new('https://www.patreon.com/HudsonThames')
    except webbrowser.Error as error:
        try:
            webbrowser.get('windows-default').open_new('https://www.patreon.com/HudsonThames')
        except webbrowser.Error as error:
            print('Support us on Patreon: https://www.patreon.com/HudsonThames')

print()
print()
print(textwrap.dedent("""\
MlFinLab needs you! We need your help to keep on maintaining and implementing academic research based on financial 
machine learning, for open-source. In order for us to continue we need to raise $4000 of monthly donations via Patreon -
by December 2020. If we can't reach our goal, we intend to leave the package as open-source but to make the documentation
a paid for service. 

Our road map for 2020 is to implement the text book: Machine Learning for Asset Managers by Marcos Lopez de Prado, 
as well as a few papers from the Journal of Financial Data Science. We are hiring a full time developer for 3 months 
to help us reach our goals. The money that you, our sponsors, contribute will go directly to paying salaries and other 
expenses such as journal subscriptions and data. 

We need your help to continue maintaining and developing this community. Thank you for using our package and we 
invite you to join our slack channel using the following link:
https://join.slack.com/t/mlfinlab/shared_invite/zt-c62u9gpz-VFc13j6da~UVg3DkV7~RjQ
"""))
print()
print()
