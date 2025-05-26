"""  Parse data from cian.ru
https://github.com/lenarsaitov/cianparser
"""
import datetime

import cianparser
import pandas as pd

moscow_parser = cianparser.CianParser(location="Москва")


def main():
    """
    Function docstring
    """
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_path = f'./data/raw/3_room.csv'
    data = moscow_parser.get_flats(
        deal_type="sale",
        rooms=(3,),
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": 56,
            "max_price": 100000000,
            "object_type": "secondary"
        })
    df = pd.DataFrame(data)

    df.to_csv(csv_path,
              encoding='utf-8',
              index=False)


if __name__ == '__main__':
    main()
