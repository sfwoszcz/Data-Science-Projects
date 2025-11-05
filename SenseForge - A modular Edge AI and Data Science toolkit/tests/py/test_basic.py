from sensorkit.pipeline import run

def test_run():
    df = run('data/sample_vibration.csv')
    assert 'rms' in df.columns
