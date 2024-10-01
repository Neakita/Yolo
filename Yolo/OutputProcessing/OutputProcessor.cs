using CommunityToolkit.Diagnostics;

namespace Yolo.OutputProcessing;

public abstract class OutputProcessor<T>
{
	public float MinimumConfidence
	{
		get => _minimumConfidence;
		set
		{
			Guard.IsInRange(value, 0, 1);
			_minimumConfidence = value;
		}
	}

	public abstract IEnumerable<T> Process(RawOutput output);

	private float _minimumConfidence = 0.3f;
}