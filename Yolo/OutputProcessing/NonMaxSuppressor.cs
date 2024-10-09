using System.Collections.ObjectModel;
using Collections.Pooled;
using CommunityToolkit.Diagnostics;

namespace Yolo.OutputProcessing;

internal sealed class NonMaxSuppressor : IDisposable
{
	public float MaximumIoU
	{
		get => _maximumIoU;
		set
		{
			Guard.IsBetween(value, 0, 1);
			_maximumIoU = value;
		}
	}

	public NonMaxSuppressor()
	{
		_buffer = new PooledList<Detection>();
		_wrappedBuffer = new ReadOnlyCollection<Detection>(_buffer);
	}

	public ReadOnlyCollection<Detection> Suppress(PooledList<Detection> detections)
	{
		PrepareBuffer();
		foreach (var detection in detections)
		{
			if (Intersects(detection, _buffer))
				continue;
			_buffer.Add(detection);
		}
		return _wrappedBuffer;
	}

	public void Dispose()
	{
		_buffer.Dispose();
	}

	private readonly ReadOnlyCollection<Detection> _wrappedBuffer;
	private readonly PooledList<Detection> _buffer;
	private float _maximumIoU = 0.45f;

	private void PrepareBuffer()
	{
		_buffer.Clear();
	}

	private bool Intersects(Detection subject, PooledList<Detection> passedSubjects)
	{
		foreach (var passedDetection in passedSubjects)
		{
			if (subject.ClassId != passedDetection.ClassId)
				continue;
			var intersection = CalculateIoU(subject.Bounding, passedDetection.Bounding);
			if (intersection > _maximumIoU)
				return true;
		}
		return false;
	}

	private static float CalculateIoU(in Bounding first, in Bounding second)
	{
		if (first.Area == 0)
			return 0;
		if (second.Area == 0)
			return 0;
		var intersection = GetIntersection(first, second);
		return intersection.Area / (first.Area + second.Area - intersection.Area);
	}

	private static Bounding GetIntersection(Bounding first, Bounding second)
	{
		float left = Math.Max(first.Left, second.Left);
		float right = Math.Min(first.Right, second.Right);
		float top = Math.Max(first.Top, second.Top);
		float bottom = Math.Min(first.Bottom, second.Bottom);
		if (right >= left && bottom >= top)
			return new Bounding(left, top, right, bottom);
		return Bounding.Empty;
	}
}