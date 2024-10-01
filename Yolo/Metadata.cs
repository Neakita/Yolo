using System.Collections.Immutable;
using CommunityToolkit.Diagnostics;
using Microsoft.ML.OnnxRuntime;

namespace Yolo;

public class Metadata
{
	public ImmutableArray<string> ClassesNames { get; }
	internal string Version { get; }
	internal Size ImageSize { get; }
	internal byte BatchSize { get; }
	internal Task Task { get; }

	internal Metadata(InferenceSession session)
	{
		var metadata = session.ModelMetadata.CustomMetadataMap;
		Version = metadata["version"];
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
	}

	private static Size ParseSize(string str)
	{
		// parse a string like [640, 640]
		var stringSpan = str.AsSpan();
		stringSpan = stringSpan[1..^1];
		Span<Range> rangesSpan = stackalloc Range[2];
		var rangesCount = stringSpan.Split(rangesSpan, ", ");
		Guard.IsEqualTo(rangesCount, 2);
		var width = ushort.Parse(stringSpan[rangesSpan[0]]);
		var height = ushort.Parse(stringSpan[rangesSpan[1]]);
		return new Size(width, height);
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