import pathlib
import pandas as pd

BASE_PATH = pathlib.Path(__file__).parent.parent.resolve()
DATA_PATH = BASE_PATH / "data"

def load_pop_data():
    csv_path = pathlib.Path(DATA_PATH /'camau.csv')
    pop_df = pd.read_csv(csv_path)
    pop_df = pop_df.set_index(['year'])

    csv_path = pathlib.Path(DATA_PATH /'camau_birth.csv')
    birth_df = pd.read_csv(csv_path)
   
    csv_path = pathlib.Path(DATA_PATH /'camau_cdr.csv')
    death_df = pd.read_csv(csv_path)

    return pop_df, birth_df, death_df


death_rates_by_age = {0: [0.04891194320512563,
  0.033918668027576884,
  0.024432727316924446,
  0.019408181689238014,
  0.018148370056426787,
  0.014535299525724738,
  0.013035498777442808,
  0.011254176500418787,
  0.00931061542295055,
  0.006118489529305104,
  0.005678373734277731,
  0.005333479069860507,
  0.004783694403203202,
  0.0043347634430972014],
 5: [0.005654833192668025,
  0.005297274566250849,
  0.0044664893264966974,
  0.003846307644222109,
  0.005107979016750706,
  0.002711430406000275,
  0.002147435231479338,
  0.0016691038222857585,
  0.0013164357375949395,
  0.0010181764308804923,
  0.0007447183497013854,
  0.0005658632755602,
  0.0005441317519979431,
  0.0005100562239187423],
 15: [0.0030517485729701097,
  0.002700755057428503,
  0.002489838533286184,
  0.002584612327589561,
  0.004088806859043201,
  0.0023927769572424534,
  0.002209771395787853,
  0.001885328047816437,
  0.0016581044683420282,
  0.0015277277213098862,
  0.001427063312059633,
  0.0012953587483924324,
  0.0012356506891999784,
  0.0011729827102219131],
 35: [0.009386682178966717,
  0.007892734017957679,
  0.006517135357865979,
  0.006260889686458326,
  0.008689175571776193,
  0.00479116361370093,
  0.004179181883878279,
  0.004093556688069908,
  0.004031166261049662,
  0.003907751109471642,
  0.0034886248607591164,
  0.003155553373359392,
  0.003040343537587886,
  0.002980208830203494],
 50: [0.1417162315914261,
  0.14004771975249025,
  0.14050114600763583,
  0.13991314779584452,
  0.1446541208480976,
  0.12199942887738785,
  0.11497764682078969,
  0.10618998428462792,
  0.09778759372213668,
  0.09191279368157758,
  0.08603053502488202,
  0.08429645911423628,
  0.08132411751977145,
  0.08108759595140565]}

death_rate_years = [1952.5,
 1957.5,
 1962.5,
 1967.5,
 1972.5,
 1977.5,
 1982.5,
 1987.5,
 1992.5,
 1997.5,
 2002.5,
 2007.5,
 2012.5,
 2017.5]