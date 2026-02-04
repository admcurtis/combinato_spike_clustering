#%% Dependencies
import argparse

import create_spike_and_event_tables
import count_spikes
import detect_concepts
import create_raster_data

#%% Main
def main(save=False, s_min=0.3, b_min=0.3):

    waveform_df, events_df = create_spike_and_event_tables.run()

    spike_counts_df = count_spikes.count_spikes_per_ppt(
        waveform_df, events_df, s_min=s_min, b_min=b_min
    )

    concepts = detect_concepts.detect_concepts(spike_data=spike_counts_df)

    raster_data = create_raster_data.create_raster_data(
        concepts, waveform_df, spike_counts_df
    )

    if save:
        waveform_df.to_csv("./all_spike_waveforms.csv", index=False)
        events_df.to_csv("all_events.csv", index=False)
        spike_counts_df.to_csv("spike_counts.csv", index=False)
        concepts.to_csv("./detected_concepts.csv", index=False)
        raster_data.to_pickle("raster_data.pkl")

    print(concepts)


#%% Run
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run spike processing and concept detection pipeline"
    )

    parser.add_argument(
        "--save", action="store_true",
        help="Save all intermediate and final outputs to disk"
    )

    parser.add_argument(
        "--smin", type=float, default=0.3,
        help="Time after stimulus onset to start counting spikes (seconds)"
    )

    parser.add_argument(
        "--bmin", type=float, default=0.3,
        help="Time before stimulus onset for baseline window (seconds)"
    )

    args = parser.parse_args()

    main(save=args.save, s_min=args.smin, b_min=args.bmin)