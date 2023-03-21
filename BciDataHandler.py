import mne
from moabb.datasets import BNCI2014001


class BciDataHandler:
    def __init__(self):
        self.data = BNCI2014001()
        self.subjects_epochs = {}
        self.subjects_labels = {}
        self.subjects_id = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.sessions_id = ['session_T', 'session_E']
        self.runs_id = ['run_0', 'run_1', 'run_2', 'run_3', 'run_4', 'run_5']
        self.events_desc = {'left_hand': 1, 'right_hand': 2, 'both_feet': 3, 'tongue': 4}
        self.tmin, self.tmax, self.t_crop = -.5, 4.5, .5
        self.selected_events = ['left_hand', 'right_hand']
        self.raw = self.data.get_data()
        self.picks = mne.pick_types(self.data.get_data()[1]['session_T']['run_0'].info, eeg=True, stim=False)

    def instantiate_dataset(self):
        for subject_id in self.subjects_id:
            print('subject_id: ', subject_id)
            epochs = []
            for session_id in self.sessions_id:
                print('session_id: ', session_id)
                for run_id in self.runs_id:
                    loop_raw = self.raw[subject_id][session_id][run_id]
                    events = mne.find_events(loop_raw, 'stim')
                    run_epochs = mne.Epochs(
                        loop_raw,
                        events,
                        self.events_desc,
                        picks=self.picks,
                        tmin=self.tmin,
                        tmax=self.tmax,
                        preload=True
                    )[self.selected_events]
                    epochs.append(run_epochs)

            self.subjects_epochs[subject_id] = (mne.concatenate_epochs(epochs)).filter(5, 60)
            self.subjects_labels[subject_id] = [event[2] for event in self.subjects_epochs[subject_id].events]
