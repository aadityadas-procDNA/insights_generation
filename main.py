from pipeline.config import *
from pipeline.data_prep import *
from pipeline.bocpd import *
from pipeline.mmm_data_prep import *
from pipeline.mmm_fit import *
from pipeline.integration import *
from pipeline.validation import *
from pipeline.insights import generate_insights
import pipeline.validation as validation


def main():
    data_prep()
    bocpd()
    mmm_data_prep()
    mmm_fit()
    integration()
    validation()
    generate_insights()

if __name__ == "__main__":
    main()
