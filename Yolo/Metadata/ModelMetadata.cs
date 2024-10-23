using System.Collections.Immutable;
using CommunityToolkit.Diagnostics;
using Microsoft.ML.OnnxRuntime;

namespace Yolo.Metadata;

public class ModelMetadata
{
	public ImmutableArray<string> ClassesNames { get; }
	public Vector2D<int> ImageSize { get; }
	public byte Version { get; }
	public byte BatchSize { get; }
	public Task Task { get; }

	internal ModelMetadata(InferenceSession session)
	{
		var metadata = session.ModelMetadata.CustomMetadataMap;
		BatchSize = byte.Parse(metadata["batch"]);
		ImageSize = ParseSize(metadata["imgsz"]);
		Task = metadata["task"] switch
		{
			"classify" => Task.Classify,
			"detect" => Task.Detect,
			"obb" => Task.Obb,
			"pose" => Task.Pose,
			"segment" => Task.Segment,
			_ => throw new ArgumentOutOfRangeException()
		};
		ClassesNames = ParseNames(metadata["names"]);
		Version = DetectVersion(session);
	}

	private byte DetectVersion(InferenceSession session)
	{
		// YOLOv10 output shape => [<batch>, 300, 6]
		if (Task == Task.Detect && session.OutputMetadata.Values.First().Dimensions[2] == 6)
			return 10;
		return 8;
	}

	private static Vector2D<int> ParseSize(string str)
	{
		// parse a string like [640, 640]
		var stringSpan = str.AsSpan();
		stringSpan = stringSpan[1..^1];
		Span<Range> rangesSpan = stackalloc Range[2];
		var rangesCount = stringSpan.Split(rangesSpan, ", ");
		Guard.IsEqualTo(rangesCount, 2);
		var width = ushort.Parse(stringSpan[rangesSpan[0]]);
		var height = ushort.Parse(stringSpan[rangesSpan[1]]);
		return new Vector2D<int>(width, height);
	}

	private static ImmutableArray<string> ParseNames(string str)
	{
		str = str[1..^1];
		var split = str.Split(", ");
		var count = split.Length;
		var builder = ImmutableArray.CreateBuilder<string>(count);
		builder.Count = count;
		foreach (var value in split)
		{
			var valueSplit = value.Split(": ");
			var id = int.Parse(valueSplit[0]);
			var name = valueSplit[1][1..^1].Replace('_', ' ');
			builder[id] = name;
		}
		return builder.DrainToImmutable();
	}
}