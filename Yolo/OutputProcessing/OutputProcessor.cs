namespace Yolo.OutputProcessing;

public interface OutputProcessor<out T>
{
	float MinimumConfidence { get; set; }
	IEnumerable<T> Process(RawOutput output);
}