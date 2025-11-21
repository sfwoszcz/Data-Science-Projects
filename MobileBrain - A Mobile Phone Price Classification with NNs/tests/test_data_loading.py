# tests/test_data_loading.pyfrom src.data_loading import load_mobilephone_data


def test_load_mobilephone_data():
    X, y, le = load_mobilephone_data("data/MobilePhone.csv")
    assert X.shape[0] == len(y)
    assert X.shape[1] > 0
