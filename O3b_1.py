from Data_Setup import sample_processing

sample_processing('H', 1258631168, 1258754047, 2, 'Spectrograms', q_range=(10, 100), frange=(20, 1200),
                  f_duration=1, show=True, method='welch')
