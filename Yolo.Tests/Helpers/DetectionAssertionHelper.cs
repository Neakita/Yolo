using FluentAssertions;

namespace Yolo.Tests.Helpers;

internal static class DetectionAssertionHelper
{
	public static void AssertPrediction(
		Metadata metadata,
		string expectedDetections,
		IReadOnlyList<Detection> actualDetections)
	{
		var expectations = ParseExpectations(expectedDetections).ToList();
		var countedDetections = actualDetections
			.GroupBy(detection => detection.ClassId)
			.Select(group => (className: metadata.ClassesNames[group.Key], count: group.Count()))
			.ToList();
		foreach (var expectation in expectations)
		{
			var detectionsCount = countedDetections.FirstOrDefault(detection => detection.className == expectation.ClassName).count;
			detectionsCount.Should().BeInRange(expectation.MinimumCount, expectation.MaximumCount);
			if (detectionsCount > 0)
				countedDetections.Remove((expectation.ClassName, detectionsCount));
		}
		countedDetections.Should().BeEmpty();
	}

	private static IEnumerable<PredictionExpectation> ParseExpectations(string expectedPrediction)
	{
		return expectedPrediction.Split(',').Select(ParseExpectation);
	}

	private static PredictionExpectation ParseExpectation(string expectation)
	{
		var split = expectation.Split(':');
		var className = split[0];
		var count = split[1];
		var countSplit = count.Split('-');
		if (countSplit.Length == 1)
		{
			var parsedCount = int.Parse(countSplit[0]);
			return new PredictionExpectation(className, parsedCount, parsedCount);
		}

		var parsedMinimum = int.Parse(countSplit[0]);
		var parsedMaximum = int.Parse(countSplit[1]);
		return new PredictionExpectation(className, parsedMinimum, parsedMaximum);
	}
}