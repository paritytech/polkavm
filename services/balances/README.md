## Balance Service
The Balance Service is designed for managing assets, enabling issuers to create and manage their assets, supporting stakers in bonding and unbonding balances, and facilitating transactions between accounts.

## Asset and Account Structure in balance Service
### Asset Storage Key
```json
{
    "asset_id": "8 bytes", // uint64 little endian 8 bytes
}
```

### Asset
```json
{
    "asset_id": "8 bytes", // uint64 little endian 8 bytes
    "issuer_key": "32 bytes",
    "min_balance": "8 bytes", // uint64 little endian 8 bytes
    "symbol": "32 bytes",
    "total_supply": "8 bytes", // uint64 little endian 8 bytes
    "decimals": "1 byte"
}
```

### Account Storage Key
```json
{
    "asset_id": "8 bytes", // uint64 little endian 8 bytes
    "account_key": "32 bytes"
}
```

### Account
```json
{
    "nonce": "8 bytes", // uint64 little endian 8 bytes
    "free": "8 bytes", // uint64 little endian 8 bytes
    "reserved": "8 bytes" // uint64 little endian 8 bytes
}
```

---

## Callable Extrinsics

Note: Each extrinsic consists of three sections: the **first 32 bytes store the public key**, **the last 64 bytes store the signature**, and the middle section contains the extrinsic data.

### Create Asset Extrinsic
```json
{
    "method_id": "4 bytes", // uint32 little endian 4 bytes
    "asset_id": "8 bytes", // uint64 little endian 8 bytes
    "issuer_key": "32 bytes",
    "min_balance": "8 bytes", // uint64 little endian 8 bytes
    "symbol": "32 bytes",
    "total_supply": "8 bytes", // uint64 little endian 8 bytes
    "decimals": "1 byte"
}
```

### Mint Extrinsic
```json
{
    "method_id": "4 bytes", // uint32 little endian 4 bytes
    "asset_id": "8 bytes", // uint64 little endian 8 bytes
    "account_key": "32 bytes",
    "amount": "8 bytes" // uint64 little endian 8 bytes
}
```

### Burn Extrinsic
```json
{
    "method_id": "4 bytes", // uint32 little endian 4 bytes
    "asset_id": "8 bytes", // uint64 little endian 8 bytes
    "account_key": "32 bytes",
    "amount": "8 bytes" // uint64 little endian 8 bytes
}
```

### Bond Extrinsic
```json
{
    "method_id": "4 bytes", // uint32 little endian 4 bytes
    "asset_id": "8 bytes", // uint64 little endian 8 bytes
    "account_key": "32 bytes",
    "amount": "8 bytes" // uint64 little endian 8 bytes
}
```

### Unbond Extrinsic
```json
{
    "method_id": "4 bytes", // uint32 little endian 4 bytes
    "asset_id": "8 bytes", // uint64 little endian 8 bytes
    "account_key": "32 bytes",
    "amount": "8 bytes" // uint64 little endian 8 bytes
}
```

### Transfer Extrinsic
```json
{
    "method_id": "4 bytes", // uint32 little endian 4 bytes
    "asset_id": "8 bytes", // uint64 little endian 8 bytes
    "sender_key": "32 bytes",
    "receiver_key": "32 bytes",
    "amount": "8 bytes" // uint64 little endian 8 bytes
}
``` 

## Usage Instructions
Take create asset extrinsic for example

### Step 1: define asset to create
```go    
asset := Asset{
    AssetID:     1984,
    Issuer:      [32]byte{},
    MinBalance:  100,
    Symbol:      [32]byte{},
    TotalSupply: 100,
    Decimals:    8,
}
copy(asset.Issuer[:], v1_bytes)
copy(asset.Symbol[:], []byte("USDT"))
```

### Step 2: Generate extrinsic and sign
```go=
extrinsicsBytes := types.ExtrinsicsBlobs{}
extrinsic := CreateAssetExtrinsic{
    method_id: create_asset_id, // 0
    asset:     asset,
}
extrinsicBytes_signed := AddEd25519Sign(extrinsic.Bytes())
extrinsicsBytes = append(extrinsicsBytes, extrinsicBytes_signed)
```

### Step 3: Send it with wp
```go=
err := core0_peers[ramdamIdx].SendWorkPackageSubmission(create_asset_workPackage, extrinsicsBytes, 0)
if err != nil {
    fmt.Printf("SendWorkPackageSubmission ERR %v\n", err)
}
```

### Step 4: Check result inside balance service's storage
```go=
ShowAssetDetail(n1, balancesServiceIndex, 1984)
```

### Result
```=
Asset ID: 1984
Asset Issuer: 0101010101010101010101010101010101010101010101010101010101010101
Asset MinBalance: 100
Asset Symbol: USDT
Asset TotalSupply: 100
Asset Decimals: 8
```
