# Cloud Pipeline — Phase 6 (Planned)

This directory will contain the AWS infrastructure for SpectrumEye's cloud tier.

## Planned Stack

```
AWS IoT Core (MQTT)
  → Kinesis Data Streams
  → Lambda (event processor)
  → DynamoDB (detections) + Timestream (RSSI time series)
  → API Gateway WebSocket
  → React Dashboard (live mode)
```

## Directory Layout (to be created)

```
cloud/
├── cdk/          AWS CDK stack — IoT Core, Kinesis, Lambda, DynamoDB, API GW
└── lambda/       Lambda function handlers
```

## Interface

The edge `aws_publisher.py` already implements the full publish interface.
When this stack is deployed, update `edge/aws_publisher.py` with the real
IoT endpoint and certificate paths — no other edge code changes needed.
