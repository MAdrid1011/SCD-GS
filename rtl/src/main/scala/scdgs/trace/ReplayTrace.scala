package scdgs.trace

import scala.io.Source
import scdgs.config.ScdGsParams

/** Parsed public trace record. Values remain strings until a hardware driver consumes them. */
final case class TraceRecord(kind: String, fields: Map[String, String])

object TraceFormat {
  private val publicShape = ScdGsParams()
  private val publicMaxFrameId = publicShape.maxFrameId
  private val publicMaxTileId = (BigInt(1) << publicShape.tileWidth) - 1
  private val publicMaxBlendOrder = publicShape.renderQueueDepth - 1
  private val required = Map(
    "FRAME" -> Set("id", "edge"),
    "STATE" -> Set("gid", "frame", "mean", "covariance", "opacity", "sh", "tier"),
    "UPDATE" -> Set("gid", "frame", "dmu", "dsigma", "do", "dsh", "n", "struct"),
    "PROJECTION" -> Set(
      "gid",
      "frame",
      "r_cam",
      "j_pi",
      "j_cov",
      "view_direction",
      "sh_basis",
      "center",
      "conic",
      "radius",
      "area",
      "opacity",
      "view_color",
      "request",
      "role",
      "tile",
      "depth"
    ),
    "EVENT" -> Set("gid", "frame", "source"),
    "REQUEST" -> Set("gid", "frame", "kind"),
    "TRAJECTORY" -> Set("gid", "frame", "pixel", "blend", "coverage", "transmittance", "color", "path"),
    "TARGET" -> Set("pixel", "frame", "rgb")
  )

  private val structuralMarkers = Set("none", "insert", "delete", "invalidate", "reset")
  private val tiers = Set("active", "resident", "candidate")
  private val requestKinds = Set("user", "training")
  private val projectionRoles = Set("prior", "current")
  private val eventSources = Set("commit", "confirmation", "render-read")
  private val trajectoryPaths = Set("full", "light")
  private val hexWord = "0x[0-9a-fA-F]{1,8}".r

  private def natural(value: String, description: String): Int = {
    require(value.matches("[0-9]+"), s"$description must be a non-negative decimal integer")
    val parsed = BigInt(value)
    require(parsed.isValidInt, s"$description exceeds the public trace integer range")
    parsed.toInt
  }

  private def frameIdentifier(value: String, description: String): Int = {
    val parsed = natural(value, description)
    require(
      BigInt(parsed) <= publicMaxFrameId,
      s"$description exceeds the ${publicShape.frameWidth}-bit public frame range"
    )
    parsed
  }

  private def tileIdentifier(value: String, description: String): Int = {
    val parsed = natural(value, description)
    require(
      BigInt(parsed) <= publicMaxTileId,
      s"$description exceeds the ${publicShape.tileWidth}-bit public tile range"
    )
    parsed
  }

  private def blendOrder(value: String, description: String): Int = {
    val parsed = natural(value, description)
    require(
      parsed <= publicMaxBlendOrder,
      s"$description exceeds the public TRU blend-order range [0, $publicMaxBlendOrder]"
    )
    parsed
  }

  private def binary32List(value: String, width: Int, description: String): Unit = {
    val words = value.split(",", -1).toSeq
    require(
      words.size == width && words.forall(word => hexWord.matches(word)),
      s"$description must contain $width binary32 hex words"
    )
  }

  private def positiveNormalBinary32(value: String, description: String): Unit = {
    require(hexWord.matches(value), s"$description must be a binary32 hex word")
    val word = java.lang.Long.parseLong(value.drop(2), 16)
    val exponent = (word >>> 23) & 0xffL
    require(
      (word & 0x80000000L) == 0L && exponent > 0L && exponent < 0xffL,
      s"$description must be a positive normal binary32 value"
    )
  }

  private def finiteBinary32(value: String, description: String): Unit = {
    require(hexWord.matches(value), s"$description must be a binary32 hex word")
    val word = java.lang.Long.parseLong(value.drop(2), 16)
    val exponent = (word >>> 23) & 0xffL
    require(exponent < 0xffL, s"$description must be a finite binary32 value")
  }

  def parseLine(line: String, number: Int): Option[TraceRecord] = {
    val trimmed = line.trim
    if (trimmed.isEmpty || trimmed.startsWith("#")) None
    else {
      val tokens = trimmed.split("\\|", -1).toSeq
      val kind = tokens.head
      require(required.contains(kind), s"trace line $number has unsupported kind $kind")
      val pairs = tokens.tail.map { token =>
        val split = token.split("=", 2)
        require(split.length == 2 && split(0).nonEmpty && split(1).nonEmpty, s"trace line $number has malformed field $token")
        split(0) -> split(1)
      }
      require(pairs.map(_._1).distinct.size == pairs.size, s"trace line $number has duplicate field")
      val fields = pairs.toMap
      require(required(kind).subsetOf(fields.keySet), s"trace line $number is missing ${required(kind).diff(fields.keySet).mkString(",")}")
      require(fields.keySet.subsetOf(required(kind)), s"trace line $number has unsupported field ${fields.keySet.diff(required(kind)).mkString(",")}")
      Some(TraceRecord(kind, fields))
    }
  }

  def read(path: String): Seq[TraceRecord] = {
    val source = Source.fromFile(path)
    try source.getLines().zipWithIndex.flatMap { case (line, index) => parseLine(line, index + 1) }.toSeq
    finally source.close()
  }

  /** Validates record values, full bundle fields, and frame sequencing before a fixture drives RTL. */
  def validate(records: Seq[TraceRecord], maxGaussians: Int): Unit = {
    require(maxGaussians > 0, "maxGaussians must be positive")
    var openFrame: Option[Int] = None
    var previousBegin = -1
    var seededGids = Set.empty[Int]
    var bootstrapPriorFrames = Map.empty[Int, Int]
    var consumedBootstrapPriors = Set.empty[Int]
    var priorSummaryFrames = Map.empty[Int, Int]
    var currentViewRequests = Map.empty[(Int, Int), String]
    var currentViewTiles = Map.empty[(Int, Int), Int]
    var updateFrames = Map.empty[Int, Int]
    var validatedUpdates = Set.empty[(Int, Int)]
    var requiredCommitUpdates = Set.empty[(Int, Int)]
    var acceptedUpdateCounts = Map.empty[Int, Int]
    var committedUpdates = Set.empty[(Int, Int)]
    var propagationEvents = Map.empty[(Int, Int), String]
    var requestedUpdates = Set.empty[(Int, Int)]
    var trainingTrajectories = Set.empty[(Int, Int)]
    var trainingTargetCounts = Map.empty[(Int, Int), Int]
    var targetCounts = Map.empty[(Int, Int), Int]
    var renderReadEvents = Set.empty[(Int, Int)]

    records.zipWithIndex.foreach { case (record, index) =>
      def numeric(field: String): Int = natural(record.fields(field), s"trace record ${index + 1} field $field")
      def gid(): Int = {
        val value = numeric("gid")
        require(value < maxGaussians, s"trace record ${index + 1} gid $value is outside [0, $maxGaussians)")
        value
      }
      def inOpenFrame(): Int = {
        val frame = frameIdentifier(record.fields("frame"), s"trace record ${index + 1} frame identifier")
        require(openFrame.contains(frame), s"trace record ${index + 1} belongs to frame $frame outside its open frame")
        frame
      }
      def words(field: String, width: Int): Unit =
        binary32List(record.fields(field), width, s"trace record ${index + 1} $field")

      record.kind match {
        case "FRAME" =>
          val frame = frameIdentifier(record.fields("id"), s"trace record ${index + 1} frame identifier")
          record.fields("edge") match {
            case "begin" =>
              require(openFrame.isEmpty, s"trace record ${index + 1} opens a frame before closing frame ${openFrame.get}")
              require(frame > previousBegin, s"trace record ${index + 1} has nonmonotonic frame identifier")
              openFrame = Some(frame)
              previousBegin = frame
            case "end" =>
              require(openFrame.contains(frame), s"trace record ${index + 1} closes frame $frame without a matching begin")
              openFrame = None
            case edge => throw new IllegalArgumentException(s"trace record ${index + 1} has invalid frame edge $edge")
          }
        case "STATE" =>
          val id = gid()
          inOpenFrame()
          words("mean", 3)
          words("covariance", 6)
          words("opacity", 1)
          words("sh", publicShape.shCoefficients)
          require(tiers(record.fields("tier")), s"trace record ${index + 1} has invalid tier")
          require(!seededGids(id), s"trace record ${index + 1} has duplicate STATE for gid $id")
          require(!updateFrames.contains(id), s"trace record ${index + 1} STATE for gid $id is not initial state")
          seededGids += id
        case "UPDATE" =>
          val id = gid()
          val frame = inOpenFrame()
          words("dmu", 3)
          words("dsigma", 6)
          words("do", 1)
          words("dsh", publicShape.shCoefficients)
          words("n", 1)
          require(structuralMarkers(record.fields("struct")), s"trace record ${index + 1} has invalid structural marker")
          require(seededGids(id), s"trace record ${index + 1} UPDATE for gid $id has no preceding STATE")
          require(priorSummaryFrames.contains(id), s"trace record ${index + 1} UPDATE for gid $id has no retained prior PROJECTION")
          require(
            priorSummaryFrames(id) < frame,
            s"trace record ${index + 1} UPDATE for gid $id requires a preceding training PROJECTION"
          )
          require(
            updateFrames.get(id).forall(_ < frame),
            s"trace record ${index + 1} UPDATE for gid $id is not strictly monotonic"
          )
          if (!updateFrames.contains(id)) {
            consumedBootstrapPriors += id
          }
          val acceptedCount = acceptedUpdateCounts.getOrElse(id, 0) + 1
          acceptedUpdateCounts += id -> acceptedCount
          if (acceptedCount < publicShape.siuWindow || record.fields("struct") != "none") {
            requiredCommitUpdates += id -> frame
          }
          updateFrames += id -> frame
          validatedUpdates += id -> frame
        case "PROJECTION" =>
          val id = gid()
          val frame = inOpenFrame()
          words("r_cam", 9)
          words("j_pi", 6)
          words("j_cov", 6)
          words("view_direction", 3)
          words("sh_basis", publicShape.shBases)
          words("center", 2)
          words("conic", 3)
          Seq("radius", "area", "opacity", "depth").foreach(field => words(field, 1))
          positiveNormalBinary32(record.fields("radius"), s"trace record ${index + 1} radius")
          finiteBinary32(record.fields("depth"), s"trace record ${index + 1} depth")
          words("view_color", 3)
          val tile = tileIdentifier(record.fields("tile"), s"trace record ${index + 1} tile")
          require(requestKinds(record.fields("request")), s"trace record ${index + 1} has invalid request kind")
          require(projectionRoles(record.fields("role")), s"trace record ${index + 1} has invalid projection role")
          if (record.fields("role") == "prior") {
            require(record.fields("request") == "training", s"trace record ${index + 1} prior PROJECTION must request training")
            require(!updateFrames.contains(id), s"trace record ${index + 1} prior PROJECTION for gid $id is not bootstrap state")
            require(
              !bootstrapPriorFrames.contains(id),
              s"trace record ${index + 1} has duplicate bootstrap prior PROJECTION for gid $id"
            )
            bootstrapPriorFrames += id -> frame
            priorSummaryFrames += id -> frame
          } else {
            require(
              validatedUpdates(id -> frame),
              s"trace record ${index + 1} current PROJECTION for gid $id, frame $frame has no preceding UPDATE"
            )
            require(
              committedUpdates(id -> frame),
              s"trace record ${index + 1} current PROJECTION for gid $id, frame $frame has no preceding commit"
            )
            require(
              !currentViewRequests.contains(id -> frame),
              s"trace record ${index + 1} has duplicate current PROJECTION for gid $id, frame $frame"
            )
            currentViewRequests += (id, frame) -> record.fields("request")
            currentViewTiles += (id, frame) -> tile
          }
        case "EVENT" =>
          val id = gid()
          val frame = inOpenFrame()
          val source = record.fields("source")
          require(eventSources(source), s"trace record ${index + 1} has invalid event source")
          if (source == "commit" || source == "confirmation") {
            require(
              validatedUpdates(id -> frame),
              s"trace record ${index + 1} $source for gid $id, frame $frame has no preceding UPDATE"
            )
            require(
              !propagationEvents.contains(id -> frame),
              s"trace record ${index + 1} has duplicate $source for gid $id, frame $frame"
            )
            require(
              source == "commit" || !requiredCommitUpdates(id -> frame),
              s"trace record ${index + 1} confirmation for gid $id, frame $frame violates SIU's required commit"
            )
            propagationEvents += (id -> frame) -> source
            if (source == "commit") { committedUpdates += id -> frame }
          } else {
            require(
              trainingTrajectories(id -> frame),
              s"trace record ${index + 1} render-read for gid $id, frame $frame has no matching training TRAJECTORY"
            )
            require(
              !renderReadEvents(id -> frame),
              s"trace record ${index + 1} has duplicate render-read for gid $id, frame $frame"
            )
            renderReadEvents += id -> frame
          }
        case "REQUEST" =>
          val id = gid()
          val frame = inOpenFrame()
          val request = record.fields("kind")
          require(requestKinds(request), s"trace record ${index + 1} has invalid request kind")
          require(
            currentViewRequests.get((id, frame)).contains(request),
            s"trace record ${index + 1} REQUEST for gid $id, frame $frame must match a same-frame current PROJECTION request"
          )
          require(
            committedUpdates(id -> frame),
            s"trace record ${index + 1} REQUEST for gid $id, frame $frame has no preceding commit"
          )
          require(
            !requestedUpdates(id -> frame),
            s"trace record ${index + 1} has duplicate REQUEST for gid $id, frame $frame"
          )
          requestedUpdates += id -> frame
        case "TRAJECTORY" =>
          val id = gid()
          val frame = inOpenFrame()
          numeric("pixel")
          blendOrder(record.fields("blend"), s"trace record ${index + 1} blend")
          words("coverage", 1)
          words("transmittance", 1)
          words("color", 3)
          require(
            trajectoryPaths(record.fields("path")),
            s"trace record ${index + 1} TRAJECTORY path must be full or light"
          )
          require(
            BigInt(frame) < publicMaxFrameId,
            s"trace record ${index + 1} TRAJECTORY must use a frame below $publicMaxFrameId to preserve TUU feedback causality"
          )
          require(
            currentViewRequests.get((id, frame)).contains("training"),
            s"trace record ${index + 1} TRAJECTORY for gid $id requires a same-frame training current PROJECTION"
          )
          require(
            currentViewTiles.get((id, frame)).contains(numeric("pixel")),
            s"trace record ${index + 1} TRAJECTORY pixel must equal its current PROJECTION tile"
          )
          require(
            committedUpdates(id -> frame),
            s"trace record ${index + 1} TRAJECTORY for gid $id, frame $frame has no preceding commit"
          )
          require(
            !trainingTrajectories(id -> frame),
            s"trace record ${index + 1} has duplicate TRAJECTORY for gid $id, frame $frame"
          )
          trainingTrajectories += id -> frame
          val targetKey = numeric("pixel") -> frame
          trainingTargetCounts += targetKey -> (trainingTargetCounts.getOrElse(targetKey, 0) + 1)
          priorSummaryFrames += id -> frame
        case "TARGET" =>
          val frame = inOpenFrame()
          val targetKey = numeric("pixel") -> frame
          targetCounts += targetKey -> (targetCounts.getOrElse(targetKey, 0) + 1)
          words("rgb", 3)
        case unknown => throw new IllegalArgumentException(s"unsupported trace record $unknown")
      }
    }
    require(openFrame.isEmpty, s"trace ends with unclosed frame ${openFrame.get}")
    bootstrapPriorFrames.keys.toSeq.sorted.find(id => !consumedBootstrapPriors(id)).foreach { id =>
      throw new IllegalArgumentException(
        s"trace bootstrap prior PROJECTION for gid $id has no matching first UPDATE"
      )
    }
    validatedUpdates.toSeq.sortBy { case (id, frame) => (frame, id) }.find(key => !propagationEvents.contains(key)).foreach {
      case (id, frame) =>
        throw new IllegalArgumentException(
          s"trace omits SIU propagation EVENT for gid $id, frame $frame"
        )
    }
    currentViewRequests.keys.toSeq.sortBy { case (id, frame) => (frame, id) }.find(key => !requestedUpdates(key)).foreach {
      case (id, frame) =>
        throw new IllegalArgumentException(
          s"trace current PROJECTION for gid $id, frame $frame has no matching REQUEST"
        )
    }
    trainingTrajectories.diff(renderReadEvents).toSeq.sortBy { case (id, frame) => (frame, id) }.headOption.foreach {
      case (id, frame) =>
        throw new IllegalArgumentException(
          s"trace omits render-read for training TRAJECTORY gid $id, frame $frame"
      )
    }
    trainingTargetCounts.toSeq.sortBy { case ((pixel, frame), _) => (frame, pixel) }.collectFirst {
      case ((pixel, frame), expected) if targetCounts.getOrElse((pixel, frame), 0) < expected =>
        (pixel, frame)
    }.foreach { case (pixel, frame) =>
      throw new IllegalArgumentException(
        s"trace omits TARGET for training TRAJECTORY pixel $pixel, frame $frame"
      )
    }
    targetCounts.toSeq.sortBy { case ((pixel, frame), _) => (frame, pixel) }.collectFirst {
      case ((pixel, frame), actual) if trainingTargetCounts.getOrElse((pixel, frame), 0) < actual =>
        (pixel, frame)
    }.foreach { case (pixel, frame) =>
      throw new IllegalArgumentException(
        s"trace has TARGET for pixel $pixel, frame $frame without a matching training TRAJECTORY"
      )
    }
  }
}

/**
 * Validates a public trace and prints only mechanism counters. RTL unit tests
 * use the same record fields; this entry point deliberately has no model,
 * dataset, experimental-result, or platform dependency.
 */
object ReplayTrace {
  val defaultMaxGaussians: Int = ScdGsParams().maxGaussians

  def main(args: Array[String]): Unit = {
    val path = args.headOption.getOrElse("../docs/fixtures/six_frame.scdgs-trace")
    val records = TraceFormat.read(path)
    TraceFormat.validate(records, maxGaussians = defaultMaxGaussians)
    val frames = records.collect { case TraceRecord("FRAME", fields) if fields("edge") == "begin" => fields("id").toInt }
    require(frames == frames.sorted.distinct, "frame begin identifiers must be monotonic and unique")
    val count = records.groupMapReduce(_.kind)(_ => 1)(_ + _).withDefaultValue(0)
    val commits = records.count(record => record.kind == "EVENT" && record.fields("source") == "commit")
    val confirmations = records.count(record => record.kind == "EVENT" && record.fields("source") == "confirmation")
    val reads = records.count(record => record.kind == "EVENT" && record.fields("source") == "render-read")
    val structural = records.count(record => record.kind == "UPDATE" && record.fields("struct") != "none")
    println(
      s"{\"frames\":${frames.size},\"stateSeeds\":${count("STATE")},\"canonicalUpdates\":${count("UPDATE")}," +
        s"\"commitEvents\":$commits,\"confirmationEvents\":$confirmations,\"renderReadEvents\":$reads,\"requests\":${count("REQUEST")}," +
        s"\"trajectories\":${count("TRAJECTORY")},\"targets\":${count("TARGET")}," +
        s"\"structuralUpdates\":$structural}"
    )
  }
}
