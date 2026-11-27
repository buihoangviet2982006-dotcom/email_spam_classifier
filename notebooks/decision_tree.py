import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / ".." / "src" / "data"
MODEL_PATH = BASE_DIR / ".." / "models" / "decision_tree.pkl"


def load_and_clean(path: Path) -> pd.DataFrame:
    """Làm sạch từng tập riêng biệt, không học thống kê từ test."""
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {path}")

    df = pd.read_csv(path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(f"{path.name} phải có hai cột text và label.")

    original_count = len(df)
    df = df[["text", "label"]].dropna(subset=["text", "label"]).copy()
    df["text"] = (
        df["text"].astype(str).str.lower()
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df = df.loc[df["text"].ne("")].copy()
    df["label"] = pd.to_numeric(df["label"], errors="raise")
    if not df["label"].isin([0, 1]).all():
        raise ValueError(f"Cột label trong {path.name} phải chỉ có 0 hoặc 1.")
    df["label"] = df["label"].astype(int)
    if df.empty:
        raise ValueError(f"{path.name} không còn mẫu hợp lệ sau làm sạch.")

    print(f"\n{path.name}: {original_count:,} mẫu ban đầu.")
    print(f"Bỏ {original_count - len(df):,} mẫu thiếu nhãn hoặc nội dung rỗng.")
    return df


def clean_training_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Loại nội dung mang nhãn mâu thuẫn và mẫu trùng trong train."""
    conflicts = df.groupby("text")["label"].transform("nunique").gt(1)
    print(f"Train: bỏ {int(conflicts.sum()):,} mẫu có nhãn mâu thuẫn.")
    df = df.loc[~conflicts].copy()
    original_count = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"Train: bỏ {original_count - len(df):,} mẫu trùng nội dung.")
    if set(df["label"].unique()) != {0, 1}:
        raise ValueError("Tập train sau làm sạch phải có cả nhãn 0 và nhãn 1.")
    return df


def build_model() -> Pipeline:
    # Chỉ fit TF-IDF trên train; khi predict, test chỉ được transform.
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            min_df=2,
            sublinear_tf=True,
        )),
        ("tree", DecisionTreeClassifier(
            criterion="gini",
            max_depth=20,
            min_samples_leaf=5,
            random_state=42,
        )),
    ])


def main() -> None:
    train_df = clean_training_duplicates(load_and_clean(DATA_DIR / "train.csv"))
    test_df = load_and_clean(DATA_DIR / "test.csv")

    # Giữ các mẫu hợp lệ trong test, không chia lại dữ liệu.
    print("\nSố mẫu theo nhãn (0 = Email thường, 1 = Spam):")
    print(pd.DataFrame({
        "train": train_df["label"].value_counts(),
        "test": test_df["label"].value_counts(),
    }).reindex([0, 1]).fillna(0).astype(int).to_string())
    overlap = int(test_df["text"].isin(set(train_df["text"])).sum())
    print(f"Số mẫu test có nội dung xuất hiện trong train: {overlap:,}")
    if overlap:
        print("Lưu ý: nội dung trùng giữa train/test có thể làm điểm đánh giá cao hơn thực tế.")

    model = build_model()
    print(f"\nĐang huấn luyện trên {len(train_df):,} mẫu train...", flush=True)
    model.fit(train_df["text"], train_df["label"])

    print(f"Đang đánh giá trên {len(test_df):,} mẫu test...", flush=True)
    y_test = test_df["label"]
    y_pred = model.predict(test_df["text"])
    print(f"\nAccuracy trên tập test: {accuracy_score(y_test, y_pred):.4%}")
    print(classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["Email thường", "Spam"], digits=4, zero_division=0,
    ))
    print("Ma trận nhầm lẫn (hàng = nhãn thật, cột = nhãn dự đoán):")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=[0, 1]),
        index=["Thật: Email thường", "Thật: Spam"],
        columns=["Dự đoán: Email thường", "Dự đoán: Spam"],
    ).to_string())

    # Lưu cả TF-IDF và cây quyết định để dùng lại, ghi đè mô hình cũ.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nĐã lưu mô hình tại: {MODEL_PATH}")


if __name__ == "__main__":
    main()

