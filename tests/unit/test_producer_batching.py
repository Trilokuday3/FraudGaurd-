from streaming.producer import select_batch


def test_select_batch_returns_batch_size_rows_and_advanced_cursor():
    pool = [{"id": i} for i in range(10)]
    batch, new_cursor = select_batch(pool, cursor=0, batch_size=3)
    assert batch == [{"id": 0}, {"id": 1}, {"id": 2}]
    assert new_cursor == 3


def test_select_batch_wraps_around_the_pool():
    pool = [{"id": i} for i in range(5)]
    batch, new_cursor = select_batch(pool, cursor=3, batch_size=4)
    assert batch == [{"id": 3}, {"id": 4}, {"id": 0}, {"id": 1}]
    assert new_cursor == 2


def test_select_batch_on_empty_pool_returns_empty_and_unchanged_cursor():
    batch, new_cursor = select_batch([], cursor=5, batch_size=3)
    assert batch == []
    assert new_cursor == 5
