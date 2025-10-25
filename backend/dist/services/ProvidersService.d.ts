import { Client } from 'pg';
import { ProviderRow } from '../models/Provider';
export declare class ProvidersService {
    private readonly db;
    constructor(db: Client);
    listProviders(status?: string, limit?: number, offset?: number): Promise<{
        providers: ProviderRow[];
        total: number;
    }>;
    createProvider(data: {
        name: string;
        type: string;
        uis: string;
        config?: Record<string, unknown>;
        schedule?: string;
    }): Promise<ProviderRow>;
    getProvider(id: string): Promise<ProviderRow>;
    updateProvider(id: string, data: Partial<Record<string, unknown>>): Promise<ProviderRow>;
    deleteProvider(id: string): Promise<void>;
}
//# sourceMappingURL=ProvidersService.d.ts.map