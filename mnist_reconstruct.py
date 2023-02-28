
from einml.train import make_train_loop
from einml.prelude import *
from einml.run_name import random_run_name
import einml
import einml.progress

def make_datasets(cfg: Box):
    """Load the MNIST dataset."""

    train_ds: Dataset = tfds.load('mnist', split='train', shuffle_files=True)
    train_ds = train_ds.map(lambda x: x['image'])
    train_ds = train_ds.map(lambda x: tf.reshape(x, (28, 28, 1)))
    train_ds = train_ds.map(lambda x: tf.cast(x, tf.float32) / 255.0* 2.0 - 1.0)
    train_ds = train_ds.map(lambda x: (x, x))
    train_ds = train_ds.shuffle(10000)
    train_ds = train_ds.batch(cfg.batch_size)
    train_ds = train_ds.repeat()
    train_ds = train_ds.enumerate()
    train_ds = train_ds.take(cfg.n_train_steps)
    train_ds = train_ds.prefetch(10)

    test_ds: Dataset = tfds.load('mnist', split='test', shuffle_files=True)
    test_ds = test_ds.map(lambda x: x['image'])
    test_ds = test_ds.map(lambda x: tf.reshape(x, (28, 28, 1)))
    test_ds = test_ds.map(lambda x: tf.cast(x, tf.float32) / 255.0 * 2.0 - 1.0)
    test_ds = test_ds.map(lambda x: (x, x))
    test_ds = test_ds.shuffle(10000)
    test_ds = test_ds.batch(cfg.test_batch_size)
    test_ds = test_ds.enumerate()
    test_ds = test_ds.prefetch(10)

    return train_ds, test_ds


if __name__ == '__main__':

    cfg = Box({
        'batch_size': 128,
        'test_batch_size': 128,
        'n_train_steps': 10000,
        'mmd_weight': 100.0,
        'mmd_kernel_r': 1.0,
        'mmd_noise_batch_size': 128,
        'learning_rate': 1e-3,
        'run_name': random_run_name(),
    })


    # model
    model_name = input("Enter model name: ")
    with einml.progress.create_progress_manager() as prog:
        with prog.enter_spinner('Load model', 'Loading model...', delete_on_success=True):
            model: Model = keras.models.load_model('models/' + model_name)

        cfg.output_dir = f"artifacts/{model_name}-{cfg.run_name}"

        # data
        with prog.enter_spinner('Load data', 'Loading data...', delete_on_success=True):
            train_ds, test_ds = make_datasets(cfg)


        optimizer=tf.keras.optimizers.Adam(cfg.learning_rate)

        # compile model
        with prog.enter_spinner('Compile model', 'Compiling model...', delete_on_success=True):
            model.compile(
                optimizer=optimizer,
                loss=model.loss_fn,
            )

        # create train loop
        with prog.enter_spinner('Create train loop', 'Creating train loop...', delete_on_success=True):
            train_loop = make_train_loop(
                model=model,
                data=train_ds,
                val_data=test_ds,
                output_dir=cfg.output_dir,
                optimizer=optimizer,
            )

        # train
        train_loop(pm=prog)
