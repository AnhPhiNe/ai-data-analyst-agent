import pytest

from backend.tools.sql_safety import validate_read_only_sql


def test_validate_read_only_sql_accepts_simple_select() -> None:
    result = validate_read_only_sql("SELECT department, salary FROM dataset", 10)

    assert result.sql == "SELECT department, salary FROM dataset"
    assert result.limit == 10
    assert result.executable_sql.endswith("LIMIT 10")


def test_validate_read_only_sql_accepts_group_by_having() -> None:
    result = validate_read_only_sql(
        """
        SELECT department, AVG(salary) AS avg_salary
        FROM dataset
        GROUP BY department
        HAVING COUNT(*) >= 2
        ORDER BY avg_salary DESC
        """,
        None,
    )

    assert result.limit == 100
    assert "GROUP BY department" in result.sql


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM dataset",
        "DROP TABLE dataset",
        "COPY dataset TO 'x.csv'",
        "SELECT * FROM read_csv('x.csv')",
        "SELECT * FROM dataset; SELECT * FROM dataset",
        "SELECT * FROM other_table",
    ],
)
def test_validate_read_only_sql_rejects_unsafe_queries(sql: str) -> None:
    with pytest.raises(ValueError):
        validate_read_only_sql(sql)


def test_validate_read_only_sql_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError):
        validate_read_only_sql("SELECT * FROM dataset", 101)
