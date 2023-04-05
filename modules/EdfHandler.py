import mne


class EdfHandler:

    @staticmethod
    def getData(filePath):
        raw = mne.io.read_raw_edf \
                (
                input_fname=filePath,
                preload=True
            )

        tmin, tmax, t_crop = -.5, 4.5, .5
        #events = ['LeftExec', 'LeftPrepare', 'NewRun', 'NewTrial', 'Resting', 'RightExec', 'RightPrepare']
        selected_events = ['1', '6'] # LeftExec e RightExec
        events, event_id = mne.events_from_annotations(raw)
        return mne.Epochs(raw, events, tmin=tmin, tmax=tmax, preload=True)[selected_events]

    @staticmethod
    def getAllData(files):
        # files = \
        #     [
        #         "C:\\Users\\davi2\Desktop\\bci\\datasets_ufjf\\bci\\001.edf",
        #         "C:\\Users\\davi2\Desktop\\bci\\datasets_ufjf\\bci\\002.edf",
        #         "C:\\Users\\davi2\Desktop\\bci\\datasets_ufjf\\bci\\003.edf",
        #         "C:\\Users\\davi2\Desktop\\bci\\datasets_ufjf\\bci\\004.edf",
        #         "C:\\Users\\davi2\Desktop\\bci\\datasets_ufjf\\bci\\005.edf",
        #         "C:\\Users\\davi2\Desktop\\bci\\datasets_ufjf\\bci\\006.edf",
        #     ]

        epochs = []
        labels = []
        for file in files:
            _epochs = EdfHandler.getData(file)
            epochs.append(_epochs)
            labels.append([event[2] for event in _epochs.events])

        return epochs, labels



