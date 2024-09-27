namespace Yolo.OutputProcessing;

internal sealed class ReverseValueDetectionComparer : IComparer<ValueDetection>
{
	public static ReverseValueDetectionComparer Instance { get; } = new();

	public int Compare(ValueDetection x, ValueDetection y)
	{
		return y.Confidence.CompareTo(x.Confidence);
	}
}