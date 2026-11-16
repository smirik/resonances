"""The resume example of docs/serialization.md (run by docs/examples/run_all.py with a short span).

Here it runs as a fresh batch run of two bodies (one per batch); after an interruption the same call resumes it,
skipping the completed batches (simulation_state.json is removed after a run that completes). Batch runs start
worker processes, so the script needs the `if __name__ == '__main__':` guard.
"""

# --8<-- [start:resume]
import resonances

asteroids = [463, 490]  # the original run's list, in the same order
settings = dict(integration_years=1_000, batch_threshold=1, batch_size=1, planets=['Jupiter', 'Saturn'], type='mmr')


def main():
    sim = resonances.find(
        asteroids=asteroids,  # the same list, in the same order
        name="my-simulation",
        date="2026-10-13",  # the date of the interrupted run
        save_path="output/resume",  # the same folder
        plot_path="output/resume-plots",
        **settings,  # the same settings as the interrupted run
    )
    sim.run(progress=True)


if __name__ == '__main__':  # a batch run starts worker processes: the guard is required
    main()
# --8<-- [end:resume]
