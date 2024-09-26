namespace Yolo;

internal sealed class ReversePredictionConfidenceComparer : IComparer<Prediction>
{
	public static ReversePredictionConfidenceComparer Instance { get; } = new();

	public int Compare(Prediction? x, Prediction? y)
	{
		if (ReferenceEquals(x, y)) return 0;
		if (y is null) return -1;
		if (x is null) return 1;
		return y.Confidence.CompareTo(x.Confidence);
	}
}