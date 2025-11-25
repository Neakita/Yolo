using TensorWeaver.OutputData;

namespace TensorWeaver.OutputProcessing;

public sealed class ReversePredictionConfidenceComparer : IComparer<Classification>
{
	public static ReversePredictionConfidenceComparer Instance { get; } = new();

	public int Compare(Classification x, Classification y)
	{
		return y.Confidence.CompareTo(x.Confidence);
	}
}