from collections import Counter

import numpy as np
import polars as pl
import psycopg2
from datasets import load_dataset


def sequential_split_train_val_test(
    df: pl.LazyFrame,
    test_timestamp: int,
    val_size: int = 0,
    gap_size: int = 0,
    drop_non_train_items: bool = False,
    engine: str = "streaming",
) -> tuple[pl.LazyFrame, pl.LazyFrame | None, pl.LazyFrame]:
    """
    Splits the dataset into training, validation, and test segments based on the provided timestamps.
    """

    def drop(df: pl.LazyFrame, unique_train_item_ids) -> pl.LazyFrame:
        if not drop_non_train_items:
            return df

        return df.select(
            "uid",
            pl.all()
            .exclude("uid")
            .list.gather(
                pl.col("item_id").list.eval(
                    pl.arg_where(pl.element().is_in(unique_train_item_ids.get_column("item_id").implode()))
                )
            ),
        ).filter(pl.col("item_id").list.len() > 0)

    train_timestamp = test_timestamp - gap_size - val_size - (gap_size if val_size != 0 else 0)

    assert gap_size >= 0
    assert val_size >= 0
    assert train_timestamp > 0

    df_lazy = df.lazy()

    train = df_lazy.select(
        "uid",
        pl.all()
        .exclude("uid")
        .list.gather(pl.col("timestamp").list.eval(pl.arg_where(pl.element() < train_timestamp))),
    ).filter(pl.col("item_id").list.len() > 0)

    unique_train_uids = train.select("uid").unique().collect(engine=engine)
    unique_train_item_ids = train.explode("item_id").select("item_id").unique().collect(engine=engine)

    validation = None
    if val_size != 0:
        validation = (
            df_lazy.select(
                "uid",
                pl.all()
                .exclude("uid")
                .list.gather(
                    pl.col("timestamp").list.eval(
                        pl.arg_where(
                            (pl.element() >= test_timestamp - val_size - gap_size)
                            & (pl.element() < test_timestamp - gap_size)
                        )
                    )
                ),
            )
            .with_columns(pl.col("uid").is_in(unique_train_uids.get_column("uid").implode()).alias("uid_in_train"))
            .filter("uid_in_train")
            .drop("uid_in_train")
        )

        validation = drop(validation, unique_train_item_ids).filter(pl.col("item_id").list.len() > 0)

    test = (
        df_lazy.select(
            "uid",
            pl.all()
            .exclude("uid")
            .list.gather(pl.col("timestamp").list.eval(pl.arg_where(pl.element() >= test_timestamp))),
        )
        .with_columns(pl.col("uid").is_in(unique_train_uids.get_column("uid").implode()).alias("uid_in_train"))
        .filter("uid_in_train")
        .drop("uid_in_train")
    )

    test = drop(test, unique_train_item_ids).filter(pl.col("item_id").list.len() > 0)

    return train, validation, test


def compute_item_statistics(dataset):
    item_counts = Counter()
    all_cnt = 0
    idx = 0
    while True:
        try:
            sample = dataset[idx]
            for item_id in sample["labels"]["item_id"]:
                item_counts[item_id] += 1
                all_cnt += 1
            idx += 1
        except:
            break

    return item_counts, all_cnt


def load_and_process_yambda(
    size="50m",
    min_seq_len=2,
    test_timestamp=None,
    val_size=None,
):
    """
    Loads Yambda dataset, preprocesses it, and returns train, valid, and eval DataFrames.
    """
    print("Loading dataset...")
    listens_data = load_dataset("yandex/yambda", data_dir=f"sequential/{size}", data_files="listens.parquet")
    yambda_df = pl.from_arrow(listens_data["train"].data.table)

    print("Filtering dataset...")
    cols = [
        "timestamp",
        "item_id",
        "is_organic",
        "played_ratio_pct",
        "track_length_seconds",
    ]
    yambda_df = (
        yambda_df.filter(pl.col("uid") % 200 == 0)
        .explode(cols)
        .filter((pl.col("is_organic") == 0) & (pl.col("played_ratio_pct") >= 50))
        .sort("timestamp")
        .group_by("uid", maintain_order=True)
        .agg([pl.col(col) for col in cols])
        .select(["uid", "timestamp", "item_id"])
    )

    print("Mapping item IDs...")
    unique_items = (
        yambda_df.select(pl.col("item_id"))
        .explode(pl.col("item_id"))
        .unique()
        .sort(by="item_id")
        .with_row_index("new_item_id", offset=1)
    )
    item_mapping = dict(zip(unique_items["item_id"], unique_items["new_item_id"]))

    yambda_df = yambda_df.with_columns(
        [
            pl.col("item_id")
            .map_elements(
                lambda items: [item_mapping[item] for item in items],
                return_dtype=pl.List(pl.UInt32),
            )
            .alias("item_id")
        ]
    )

    print("Splitting dataset...")
    train_events_df, valid_events_df, eval_events_df = sequential_split_train_val_test(
        yambda_df.lazy(),
        test_timestamp=test_timestamp,
        val_size=val_size,
        gap_size=0,
        drop_non_train_items=False,
    )

    train_events_df = train_events_df.collect()
    valid_events_df = valid_events_df.collect()
    eval_events_df = eval_events_df.collect()

    print("Joined events...")
    joined_events_df = train_events_df.join(
        valid_events_df,
        on="uid",
        how="left",
        suffix="_valid",
    ).join(
        eval_events_df,
        on="uid",
        how="left",
        suffix="_test",
    )

    print("Creating incremental sequences...")
    train_data = train_events_df.filter(pl.col("item_id").list.len() >= min_seq_len).sort(by="uid")

    valid_data = (
        joined_events_df.filter(pl.col("item_id_valid").is_not_null())
        .with_columns(k=pl.int_ranges(1, pl.col("item_id_valid").list.len() + 1))
        .explode("k")
        .with_columns(
            item_id=pl.concat_list([pl.col("item_id"), pl.col("item_id_valid").list.slice(0, pl.col("k"))]),
            timestamp=pl.concat_list(
                [
                    pl.col("timestamp"),
                    pl.col("timestamp_valid").list.slice(0, pl.col("k")),
                ]
            ),
        )
        .filter(pl.col("item_id").list.len() >= min_seq_len)
        .sort(["uid", "k"])
        .select(["uid", "timestamp", "item_id"])
    )

    empty_list = pl.lit([], dtype=pl.List(pl.UInt32))

    eval_data = (
        joined_events_df.filter(pl.col("item_id_test").is_not_null())
        .with_columns(
            item_id_valid_filled=pl.when(pl.col("item_id_valid").is_null())
            .then(empty_list)
            .otherwise(pl.col("item_id_valid")),
            timestamp_valid_filled=pl.when(pl.col("timestamp_valid").is_null())
            .then(empty_list)
            .otherwise(pl.col("timestamp_valid")),
        )
        .with_columns(
            train_valid_item_id=pl.concat_list([pl.col("item_id"), pl.col("item_id_valid_filled")]),
            train_valid_timestamp=pl.concat_list([pl.col("timestamp"), pl.col("timestamp_valid_filled")]),
            k=pl.int_ranges(1, pl.col("item_id_test").list.len() + 1),
        )
        .explode("k")
        .with_columns(
            item_id=pl.concat_list(
                [
                    pl.col("train_valid_item_id"),
                    pl.col("item_id_test").list.slice(0, pl.col("k")),
                ]
            ),
            timestamp=pl.concat_list(
                [
                    pl.col("train_valid_timestamp"),
                    pl.col("timestamp_test").list.slice(0, pl.col("k")),
                ]
            ),
        )
        .filter(pl.col("item_id").list.len() >= min_seq_len)
        .sort(["uid", "k"])
        .select(["uid", "timestamp", "item_id"])
    )

    return train_data, valid_data, eval_data, item_mapping


def load_and_process_db(
    db_url: str,
    min_seq_len: int = 2,
    val_size: int = 0,
    test_timestamp: int = 0,
):
    """
    Loads data from Postgres interactions table.
    Returns: train_data, valid_data, item_mapping
    """
    print("Loading data from Postgres...")

    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, item_id, created_at FROM interactions")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        raise ValueError("No data in interactions table")

    processed_rows = []
    user_map = {}
    next_uid = 0

    print(f"Fetched {len(rows)} rows. Processing...")

    for r in rows:
        u_str, i_val, ts = r
        if u_str not in user_map:
            user_map[u_str] = next_uid
            next_uid += 1

        ts_val = int(ts.timestamp())
        processed_rows.append((user_map[u_str], i_val, ts_val))

    df = pl.DataFrame(processed_rows, schema=["uid", "item_id", "timestamp"], orient="row")

    print("Mapping item IDs...")
    unique_items = df.select(pl.col("item_id")).unique().sort(by="item_id").with_row_index("new_item_id")
    item_mapping = dict(zip(unique_items["item_id"], unique_items["new_item_id"]))

    df = df.with_columns(pl.col("item_id").replace(item_mapping, default=None).cast(pl.UInt32).alias("item_id"))

    df_grouped = df.sort("timestamp").group_by("uid", maintain_order=True).agg([pl.col("timestamp"), pl.col("item_id")])

    print("Splitting dataset...")

    if test_timestamp == 0:
        all_ts = df["timestamp"].to_list()
        limit = np.percentile(all_ts, 80)
        test_timestamp = int(limit)
        val_size = int((np.max(all_ts) - np.min(all_ts)) * 0.1)

        if val_size == 0:
            val_size = 3600

    train_events_df, valid_events_df, eval_events_df = sequential_split_train_val_test(
        df_grouped.lazy(), test_timestamp=test_timestamp, val_size=val_size, gap_size=0, drop_non_train_items=False
    )

    train_events_df = train_events_df.collect()
    valid_events_df = valid_events_df.collect()
    eval_events_df = eval_events_df.collect()

    joined_events_df = train_events_df.join(
        valid_events_df,
        on="uid",
        how="left",
        suffix="_valid",
    ).join(
        eval_events_df,
        on="uid",
        how="left",
        suffix="_test",
    )

    train_data = train_events_df.filter(pl.col("item_id").list.len() >= min_seq_len).sort(by="uid")

    valid_data = (
        joined_events_df.filter(pl.col("item_id_valid").is_not_null())
        .with_columns(k=pl.int_ranges(1, pl.col("item_id_valid").list.len() + 1))
        .explode("k")
        .with_columns(
            item_id=pl.concat_list([pl.col("item_id"), pl.col("item_id_valid").list.slice(0, pl.col("k"))]),
            timestamp=pl.concat_list(
                [
                    pl.col("timestamp"),
                    pl.col("timestamp_valid").list.slice(0, pl.col("k")),
                ]
            ),
        )
        .filter(pl.col("item_id").list.len() >= min_seq_len)
        .sort(["uid", "k"])
        .select(["uid", "timestamp", "item_id"])
    )

    return train_data, valid_data, None, item_mapping
