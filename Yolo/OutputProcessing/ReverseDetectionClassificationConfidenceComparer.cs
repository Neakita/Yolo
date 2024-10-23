using Yolo.OutputData;

namespace Yolo.OutputProcessing;

internal sealed class ReverseDetectionClassificationConfidenceComparer : IComparer<Detection>
{
	public static ReverseDetectionClassificationConfidenceComparer Instance { get; } = new();

	public int Compare(Detection x, Detection y)
	{
		return y.Classification.Confidence.CompareTo(x.Classification.Confidence);
	}
}