using System.Collections.Immutable;
using Microsoft.ML.OnnxRuntime;

namespace TensorWeaver.Yolo;

public sealed class YoloMetadata
{
	public static YoloMetadata Parse(InferenceSession session)
	{
		var metadata = session.ModelMetadata.CustomMetadataMap;
		var dimensions = session.InputMetadata.Values.Single().Dimensions;
		return new YoloMetadata
		{
			ClassesNames = ParseNames(metadata["names"]),
			ImageSize = new Vector2D<int>(dimensions[3], dimensions[2]),
			Version = DetectVersion(session),
			Task = ParseTask(metadata)
		};
	}

	private static YoloTask ParseTask(Dictionary<string, string> metadata)
	{
		return metadata["task"] switch
		{
			"classify" => YoloTask.Classify,
			"detect" => YoloTask.Detect,
			"obb" => YoloTask.Obb,
			"pose" => YoloTask.Pose,
			"segment" => YoloTask.Segment,
			_ => throw new ArgumentOutOfRangeException()
		};
	}

	private static byte DetectVersion(InferenceSession session)
	{
		var task = ParseTask(session.ModelMetadata.CustomMetadataMap);
		// YOLOv10 output shape => [<batch>, 300, 6]
		if (task == YoloTask.Detect && session.OutputMetadata.Values.First().Dimensions[2] == 6)
			return 10;
		return 8;
	}

	private static ImmutableArray<string> ParseNames(string str)
	{
		str = str[1..^1];
		var split = str.Split(", ");
		var count = split.Length;
		// ReSharper disable once CollectionNeverUpdated.Local
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

	public ImmutableArray<string> ClassesNames { get; init; }
	public Vector2D<int> ImageSize { get; init; }
	public byte Version { get; init; }
	public YoloTask Task { get; init; }
}