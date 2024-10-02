namespace Yolo.OutputProcessing;

public interface OutputProcessor<out T>
{
	float MinimumConfidence { get; set; }
	IReadOnlyList<T> Process(RawOutput output);
}